# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :BiLSTM_CRF
# @File     :dataload
# @Date     :2021/7/12 21:18
# @Author   :huangjie
# @Email    :728155808@qq.com
# @Software :PyCharm
-------------------------------------------------
"""

from torch.utils.data import DataLoader,Dataset
import json
import torch
import numpy as np

class MyDataset(Dataset):
    def __init__(self,con,data_fn):
        self.con = con
        self.rel2id = json.load(open(self.con.schemas,"r",encoding="utf-8"))[0]
        self.vocab = json.load(open(self.con.vocab,"r",encoding="utf-8"))
        self.data = json.load(open(data_fn,"r",encoding="utf-8"))
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ins_json_data = self.data[idx]
        text = ins_json_data["text"]
        tags = ins_json_data["tags"]
        token_len = len(text)
        token_ids,tag_ids = [],[]
        for char in list(text):
            if char in self.vocab:
                token_ids.append(self.vocab[char])
            else:
                token_ids.append(self.vocab["unk"])
        for tag in tags:
            if tag in self.rel2id:
                tag_ids.append(self.rel2id[tag])
            else:
                tag_ids.append(self.rel2id["O"])
        mask = np.array([1]*token_len)
        token_ids = np.array(token_ids)
        tag_ids = np.array(tag_ids)
        entity_list = self.__get_entity(text,tags)
        return token_len,text,entity_list,token_ids,mask,tag_ids

    def __get_entity(self,sentence,tags):
        start = -0.1
        entity_list = []
        for k,v in enumerate(tags):
            if v.startswith("B-"):
                assert start == -0.1
                start = k
            elif v.startswith("E-"):
                assert start != -0.1
                entity_list.append(str(start)+"/"+str(k)+"/"+v.split("-")[1]+"@"+"".join(sentence[start:k+1]))
                start = -0.1
            elif v.startswith("S-"):
                assert start == -0.1
                entity_list.append(str(k)+"/"+str(k)+"/"+v.split("-")[1]+"@"+"".join(sentence[k]))
                start = -0.1
        return entity_list

def collate_fn(batch):
    token_len, text, entity_list, token_ids, mask, tag_ids = zip(*batch)
    cur_batch = len(batch)
    max_text_lenth = max(token_len)

    batch_token_ids = torch.LongTensor(cur_batch,max_text_lenth).zero_()
    batch_mask = torch.ByteTensor(cur_batch,max_text_lenth).zero_()
    batch_tag_ids = torch.LongTensor(cur_batch,max_text_lenth).zero_()

    for i in range(cur_batch):
        batch_token_ids[i,:token_len[i]].copy_(torch.from_numpy(token_ids[i]))
        batch_mask[i,:token_len[i]].copy_(torch.from_numpy(mask[i]))
        batch_tag_ids[i,:token_len[i]].copy_(torch.from_numpy(tag_ids[i]))

    return {"input_ids":batch_token_ids,
            "mask":batch_mask,
            "tag_ids":batch_tag_ids,
            "token":text,
            "entity_list":entity_list}

if __name__ == '__main__':
    file = '../dataset/dev_data.json'
    from config.config import config
    con = config()
    dataset = MyDataset(con,file)
    dataload = DataLoader(dataset,batch_size=2,collate_fn=collate_fn)
    for i in dataload:
        print(i["token_ids"])
        print(i["mask"])
