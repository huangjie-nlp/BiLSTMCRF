# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :BiLSTM_CRF
# @File     :BiLSTM_CRF
# @Date     :2021/7/17 11:30
# @Author   :huangjie
# @Email    :728155808@qq.com
# @Software :PyCharm
-------------------------------------------------
"""
import torch
import torch.nn as nn
from torchcrf import CRF

class BiLSTM_CRF(nn.Module):
    def __init__(self,con):
        super(BiLSTM_CRF, self).__init__()
        self.con = con
        self.embedding = nn.Embedding(self.con.vocab_size,self.con.word_dim)
        self.bilstm = nn.LSTM(self.con.word_dim,self.con.units,num_layers=1,bidirectional=True,batch_first=True)
        self.linear = nn.Linear(self.con.units * 2 ,self.con.num_rel)
        self.crf = CRF(self.con.num_rel,batch_first=True)

    def forward(self,token_ids,mask=None,tag_ids=None):
        embed = self.embedding(token_ids)
        lstm,(c,h) = self.bilstm(embed)
        logits = self.linear(lstm)
        if tag_ids is not None:
            logits = -1.0 * self.crf(emissions=logits,tags = tag_ids,mask = mask)
        return logits
