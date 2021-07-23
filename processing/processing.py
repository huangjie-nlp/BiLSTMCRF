# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :BiLSTM_CRF
# @File     :processing
# @Date     :2021/7/12 20:59
# @Author   :huangjie
# @Email    :728155808@qq.com
# @Software :PyCharm
-------------------------------------------------
"""
import json

def read_data(fn):
    all_data = []
    with open(fn,"r",encoding="utf-8") as f:
        lines = f.readlines()
        s = ""
        bio = []
        for line in lines:
            if line == "\n":
                all_data.append({"text":s,"tags":bio})
                s = ""
                bio = []
                continue
            char,tag = line.strip("\n").split(" ")
            s += char
            bio.append(tag)
        return all_data

def generate_scheams(data):
    rel2id = {"O":0}
    id2rel = {0:"O"}
    for i in data:
        for t in i["tags"]:
            if t not in rel2id:
                rel2id[t] = len(rel2id)
                id2rel[len(id2rel)] = t
    return rel2id,id2rel

def get_vocab(train_json):
    vocab = {"pad":0,"unk":1}
    data = json.load(open(train_json,"r",encoding="utf-8"))
    for i in data:
        text = i["text"]
        for char in list(text):
            if char not in vocab:
                vocab[char] = len(vocab)
    return vocab
if __name__ == '__main__':
    file = "../data/test.char.bmes"
    data = get_vocab("../dataset/train_data.json")
    json.dump(data,open("../dataset/vocab.json","w",encoding="utf-8"),indent=4,ensure_ascii=False)

