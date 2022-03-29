# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :BiLSTM_CRF
# @File     :train
# @Date     :2021/7/17 11:44
# @Author   :huangjie
# @Email    :728155808@qq.com
# @Software :PyCharm
-------------------------------------------------
"""
from models.BiLSTM_CRF import BiLSTM_CRF
from config.config import config
from data_loader.dataload import MyDataset,collate_fn
from torch.utils.data import DataLoader
import torch
import time
from evaluate.evaluate import evaluate
from Logger.logger import Logger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
con = config()

train_file = "dataset/train_data.json"
dev_file = "dataset/dev_data.json"

dataset = MyDataset(con,train_file)
dev_dataset = MyDataset(con,dev_file)

train_dataload = DataLoader(dataset,batch_size=con.batch_size,shuffle=True,collate_fn=collate_fn,pin_memory=True)
dev_dataload = DataLoader(dev_dataset,batch_size=1,collate_fn=collate_fn,pin_memory=True)

logger = Logger(con.log)

model = BiLSTM_CRF(con).to(device)
optimier = torch.optim.AdamW(model.parameters(),lr=0.001)

best_f1 = -1
best_epoch = -1
global_step = 0
for epoch in range(con.epoch):
    epoch_loss = 0
    epoch_init_time = time.time()
    for data in train_dataload:
        loss = model(data["token_ids"].to(device),data["mask"].to(device),data["tag_ids"].to(device))
        model.zero_grad()
        loss.backward()
        optimier.step()
        epoch_loss += loss.item()
    if (epoch+1) % 5 == 0:
        precision,recall,f1_score = evaluate(model,dev_dataload,device,con.val_resl)
        if f1_score > best_f1:
            best_f1 = f1_score
            best_epoch = epoch
            torch.save(model.state_dict(),con.save_model)
        # print("epoch:{:3d} precision:{:5.4f},recall{:5.4f},f1_score:{:5.4f},best_f1:{:5.4f},best_epoch:{:3d}".format(epoch,precision,recall,f1_score,best_f1,best_epoch))
        logger.logger.info("epoch:{:3d} precision:{:5.4f},recall{:5.4f},f1_score:{:5.4f},best_f1:{:5.4f},best_epoch:{:3d}".format(epoch,precision,recall,f1_score,best_f1,best_epoch))
    # print("epoch{} train cost {:5.2f}s epoch_loss:{:5.4f}".format(epoch,time.time()-epoch_init_time,epoch_loss))
    logger.logger.info("epoch:{} train cost {:5.2f}s epoch_loss:{:5.4f}".format(epoch,time.time()-epoch_init_time,epoch_loss))