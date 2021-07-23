# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :BiLSTM_CRF
# @File     :test
# @Date     :2021/7/18 0:10
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
con = config()

test_file = "dataset/test_data.json"


dataset = MyDataset(con,test_file)
dataload = DataLoader(dataset,batch_size=1,collate_fn=collate_fn,pin_memory=True)

model = BiLSTM_CRF(con).to(device)
model.load_state_dict(torch.load(con.save_model))
init_time = time.time()
precision, recall, f1_score = evaluate(model,dataload,device,con.test_resl)
print("test cost {:5.2f}s".format(time.time()-init_time))
print("precision:{:5.2f}, recall{:5.2f}, f1_score:{:5.2f}".format(precision,recall,f1_score))