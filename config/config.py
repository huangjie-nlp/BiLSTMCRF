# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :BiLSTM_CRF
# @File     :config
# @Date     :2021/7/12 20:59
# @Author   :huangjie
# @Email    :728155808@qq.com
# @Software :PyCharm
-------------------------------------------------
"""

class config():
    def __init__(self):
        self.num_rel = 28
        self.vocab_size = 1794
        self.word_dim = 128
        self.cuda = 0
        self.learning_rate = 1e-3
        self.units = 128
        self.batch_size = 16
        self.epoch = 100
        self.step = 100
        self.save_model = "checkpoint/bilstm_crf.pth"
        self.val_resl = "val_result/val_resl.json"
        self.test_resl = "test_result/test_resl.json"
        self.log = "./log/{}_log.log"
        self.train_fn = "dataset/train_data.json"
        self.dev_fn = "dataset/dev_data.json"
        self.test_fn = "dataset/test_data.json"
        self.schemas = "dataset/schemas.json"
        self.vocab = "dataset/vocab.json"
