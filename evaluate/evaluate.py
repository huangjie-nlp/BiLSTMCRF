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

def evaluate(model,dataload,device,fn):
    model.eval()
    gold_num,correct_num,predict_num = 0,0,0
    id2rel = json.load(open("dataset/schemas.json","r",encoding="utf-8"))[1]
    predict = []
    for data in dataload:
        resl = []
        entity_list = data["entity_list"][0]
        pred = model(data["token_ids"].to(device))
        pred_ids = model.crf.decode(pred,mask=data["mask"].to(device))
        for i in pred_ids[0]:
            resl.append(id2rel[str(i)])
        ans = parse(resl,data["token"][0])
        gold_num += len(entity_list)
        correct_num += len(set(entity_list) & set(ans))
        predict_num += len(ans)
        predict.append({"text":data["token"][0],
                        "gold":entity_list,
                        "predict":ans,
                        "new":list(set(ans) - set(entity_list)),
                        "lack":list(set(entity_list) - set(ans))})
    print("correct_num:{} predict_num:{} gold_num:{}".format(correct_num,predict_num,gold_num))
    json.dump(predict,open(fn,"w",encoding="utf-8"),indent=4,ensure_ascii=False)
    precision = correct_num / (predict_num + 1e-10)
    recall = correct_num / (gold_num + 1e-10)
    f1_score = 2 * precision * recall / (precision + recall + 1e-10)
    model.train()
    return precision,recall,f1_score