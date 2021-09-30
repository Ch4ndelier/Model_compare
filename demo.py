import torch
import logging
import math
import json
from torch.autograd import Variable
import torchvision
import numpy as np
import os
import pickle
import torch.autograd as autograd
import torch.nn.utils as utils
#from pytorch2caffe import pytorch2caffe, plot_graph

# 设置log
logger = logging.getLogger()
logger.setLevel(logging.INFO)   # 设置打印级别
formatter = logging.Formatter('%(asctime)s %(filename)s %(funcName)s [line:%(lineno)d] %(levelname)s %(message)s')

# 设置屏幕打印的格式
sh = logging.StreamHandler()
sh.setFormatter(formatter)
logger.addHandler(sh)
# 设置log保存
fh = logging.FileHandler("demo.log", encoding='utf8')
fh.setFormatter(formatter)
logger.addHandler(fh)


np.set_printoptions(threshold=np.inf)
os.environ['CUDA_VISIBLE_DEVICES'] = '6'

# caffemodel_dir = 'demo'
# model_def = os.path.join(caffemodel_dir, 'model.prototxt')
# model_weights = os.path.join(caffemodel_dir, 'model.caffemodel')


model = torch.load('/Users/liujunyuan/Model_cmp/models/13conv.pth')
#print(model)

linecount = 0
keys = model.keys()
values = model.values()
G_list = []
V_list = []

for key, value in zip(keys, values):
    if key.endswith('g'):
        G_list.append((key, value))
    elif key.endswith('v'):
        V_list.append((key, value))

if len(G_list) != len(V_list):
    print("error")
    exit()

for key, value in zip(keys, values):
    linecount += 1
    print("line:", linecount)
    print("name:", key)
    print("value size:", value.size())
    if key.endswith('g'):
        sum = 0
        a = value.reshape(1, -1)
        print(a)
        for i in a[0]:
            print(i)
            sum += math.pow(i, 2)
        print(sum)
    #print('value=', value)
    a = value.cpu().numpy()
    a = value.reshape(1, -1)
    #logging.info('{}test'.format(a))
print("linecount:", linecount)
