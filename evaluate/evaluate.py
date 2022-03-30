# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :BiLSTM_CRF
# @File     :evaluate
# @Date     :2021/7/17 23:26
# @Author   :huangjie
# @Email    :728155808@qq.com
# @Software :PyCharm
-------------------------------------------------
"""
import json
import torch
from models.BiLSTM_CRF import BiLSTM_CRF


def parse(pred_ids,token):
    start = -0.1
    resl = []
    for k,i in enumerate(pred_ids):
        if i.startswith("B-"):
            if start != -0.1:
                continue
            start = k
        elif i.startswith("E-"):
            if start == -0.1:
                continue
            resl.append(str(start)+"/"+str(k)+"/"+i.split("-")[1]+"@"+"".join(token[start:k+1]))
            start = -0.1
        elif i.startswith("S-"):
            if start != -0.1:
                continue
            resl.append(str(k)+"/"+str(k)+"/"+i.split("-")[1]+"@"+"".join(token[k]))
    return resl

class Inference():
    def __init__(self,config):
        self.config = config
        self.model = BiLSTM_CRF(self.config)
        self.device = torch.device("cuda:%d"%self.config.cuda if torch.cuda.is_available() else "cpu")
        self.model.load_state_dict(torch.load(self.config.save_model,map_location=self.device))
        self.model.to(self.device).eval()
        self.id2label = json.load(open(self.config.schemas, "r", encoding="utf-8"))[1]
        self.vocab = json.load(open(self.config.vocab,"r",encoding="utf-8"))

    def __data_porcess(self,sentence):
        sentence2id = []
        for char in sentence:
            if char in self.vocab:
                sentence2id.append(self.vocab[char])
            else:
                sentence2id.append(self.vocab["unk"])
        token_len = len(sentence2id)
        mask = [1] * token_len
        return torch.LongTensor([sentence2id]),torch.LongTensor([mask])

    def predict(self,sentence):
        input_ids,mask = self.__data_porcess(sentence)
        pred = self.model(input_ids.to(self.device),mask.to(self.device))
        decode = self.model.crf.decode(pred)
        tag_idx = []

        for idx in decode[0]:
            tag_idx.append(self.id2label[str(idx)])
        resl = parse(tag_idx,sentence)
        print(json.dumps({"sentence":sentence,"entity":resl},indent=4,ensure_ascii=False))
