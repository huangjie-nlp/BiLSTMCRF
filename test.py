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
from config.config import config
from evaluate.evaluate import Inference

config = config()
inference = Inference(config)
while True:
    sentence = input("句子:")
    predict = inference.predict(sentence)



