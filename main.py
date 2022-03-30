from framework.Framework import Framework
from config.config import config
import torch
import numpy as np

seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

config = config()
fw = Framework(config)
fw.train()
print("="*50+"test"+"="*50)
recall, precision, f1_score, predict = fw.test()
print('recall:{:5.4f} precision:{:5.4f} f1_score:{:5.4f}'.format(recall,precision,f1_score))
