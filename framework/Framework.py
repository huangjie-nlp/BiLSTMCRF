from data_loader.dataload import MyDataset,collate_fn
from models.BiLSTM_CRF import BiLSTM_CRF
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from logger.logger import Logger
import json
from evaluate.evaluate import parse
from datetime import datetime

class Framework():
    def __init__(self,config):
        self.config = config
        self.device = torch.device("cuda:%d"%self.config.cuda if torch.cuda.is_available() else "cpu")
        self.id2label = json.load(open(self.config.schemas, "r", encoding="utf-8"))[1]
        self.logger = Logger(self.config.log.format(datetime.now().strftime("%Y-%m-%d - %H:%M:%S")))

    def train(self):
        print("load data......")
        train_dataset = MyDataset(self.config,self.config.train_fn)
        dev_dataset = MyDataset(self.config,self.config.dev_fn)
        train_dataloader = DataLoader(train_dataset,batch_size=self.config.batch_size,
                                      shuffle=True,pin_memory=True,collate_fn=collate_fn,
                                      num_workers=4)
        dev_dataloader = DataLoader(dev_dataset,batch_size=1,collate_fn=collate_fn,
                                    pin_memory=True,num_workers=4)
        print("train_sentence_num:{} dev_sentence_num:{}".format(len(train_dataloader)*self.config.batch_size,
                                                                 len(dev_dataloader)))
        model = BiLSTM_CRF(self.config).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(),lr=self.config.learning_rate)

        global_step = 0
        global_loss = 0
        best_f1 = 0
        precision = 0
        recall = 0
        best_epoch = 0
        for epoch in range(self.config.epoch):
            print("[{}/{}]".format(epoch+1,self.config.epoch))
            for data in tqdm(train_dataloader):
                input_ids = data["input_ids"].to(self.device)
                mask = data["mask"].to(self.device)
                tag_ids = data["tag_ids"].to(self.device)
                loss = model(input_ids,mask,tag_ids)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                global_loss += loss.item()

                if (global_step+1) % self.config.step == 0:
                    self.logger.logger.info("epoch:{} global_step:{} global_loss:{:5.4f}".format(epoch+1,global_step+1,global_loss))
                    global_loss = 0
                global_step += 1

            r, p, f1_score, predict = self.evaluate(model,dev_dataloader)
            if f1_score > best_f1:
                best_f1 = f1_score
                precision = p
                recall = r
                best_epoch = epoch+1
                json.dump(predict,open(self.config.val_resl,"w",encoding="utf-8"),indent=4,ensure_ascii=False)
                print("epoch {} save model......".format(epoch+1))
                torch.save(model.state_dict(),self.config.save_model)
            self.logger.logger.info("epoch:{} recall:{:5.4f} precision:{:5.4f} f1_score:{:5.4f} best_f1:{:5.4f} best_epoch:{}".
                                    format(epoch+1,recall,precision,f1_score,best_f1,best_epoch))
    def evaluate(self,model, dataload):
        model.eval()
        predict_num,gold_num,correct_num = 0,0,0

        predict = []
        with torch.no_grad():
            for data in dataload:
                tags_id = []
                pred = model(data["input_ids"].to(self.device),data["mask"].to(self.device))
                decode = model.crf.decode(pred)
                for ids in decode[0]:
                    tags_id.append(self.id2label[str(ids)])
                token = data["token"][0]
                entity_list = data["entity_list"][0]
                resl = parse(tags_id,token)
                predict_num += len(set(resl))
                gold_num += len(set(entity_list))
                correct_num += len(set(entity_list) & set(resl))
                predict.append({"text": token,"gold": entity_list,"predict": resl,
                                "new": list(set(resl) - set(entity_list)),
                                "lack": list(set(entity_list) - set(resl))})
        print("predict_num:{}, gold_num:{}, correct_num:{}".format(predict_num,gold_num,correct_num))
        recall = correct_num / (gold_num + 1e-10)
        precision = correct_num / (predict_num + 1e-10)
        f1_score = 2 * recall * precision / (recall + precision + 1e-10)
        model.train()
        return recall,precision,f1_score,predict

    def test(self):
        model = BiLSTM_CRF(self.config)
        model.load_state_dict(torch.load(self.config.save_model,map_location=self.device))
        model.eval()
        model.to(self.device)

        predict_num, gold_num, correct_num = 0, 0, 0
        predict = []

        test_dataset = MyDataset(self.config,self.config.test_fn)
        dataloader = DataLoader(test_dataset,batch_size=1,collate_fn=collate_fn,pin_memory=True)

        for data in tqdm(dataloader):
            pred = model(data["input_ids"].to(self.device),data["mask"].to(self.device))
            decode = model.crf.decode(pred)
            tag_idx = []
            for idx in decode[0]:
                tag_idx.append(self.id2label[str(idx)])
            token = data["token"][0]
            resl = parse(tag_idx,token)
            entity_list = data["entity_list"][0]
            predict_num += len(set(resl))
            gold_num += len(set(entity_list))
            correct_num += len(set(resl) & set(entity_list))

            predict.append({"text": token, "gold": entity_list, "predict": resl,
                            "new": list(set(resl) - set(entity_list)),
                            "lack": list(set(entity_list) - set(resl))})
        print("predict_num:{}, gold_num:{}, correct_num:{}".format(predict_num,gold_num,correct_num))
        recall = correct_num / (gold_num + 1e-10)
        precision = correct_num / (predict_num + 1e-10)
        f1_score = 2 * recall * precision / (recall + precision + 1e-10)

        return recall, precision, f1_score, predict
