import torch
import logging
from matplotlib import pyplot as plt
import math
import json
from torch.autograd import Variable
import torchvision
import numpy as np
import os
import pickle
import torch.autograd as autograd
import torch.nn.utils as utils
from matplotlib.font_manager import FontProperties

# caffemodel_dir = 'demo'
# model_def = os.path.join(caffemodel_dir, 'model.prototxt')
# model_weights = os.path.join(caffemodel_dir, 'model.caffemodel')

# 设置plt

with open("config/cmp.json") as f:
    config = json.load(f)
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

model_x_path = config['model_x_path']
model_y_path = config['model_y_path']
model_x = torch.load(model_x_path)
model_y = torch.load(model_y_path)
model_x_name = model_x_path.split("/")[-1]
model_y_name = model_y_path.split("/")[-1]

# 设置log
logger = logging.getLogger()
logger.setLevel(logging.INFO)   # 设置打印级别
formatter = logging.Formatter('%(asctime)s %(filename)s %(funcName)s [line:%(lineno)d] %(levelname)s %(message)s')

# 设置屏幕打印的格式
sh = logging.StreamHandler()
sh.setFormatter(formatter)
logger.addHandler(sh)

# 设置log保存
logname = model_x_name + "_with_" + model_y_name + ".log"
if not os.path.exists("log"):
    os.makedirs("log")
fh = logging.FileHandler(os.path.join("log", logname), encoding='utf8')
fh.setFormatter(formatter)
logger.addHandler(fh)


np.set_printoptions(threshold=np.inf)
os.environ['CUDA_VISIBLE_DEVICES'] = '6'

#print(model)
def get_weights(model):
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

    weight_list = []

    for i in range(len(G_list)):
        g_key = G_list[i][0]
        g_val = G_list[i][1]
        print(g_val)
        v_key = V_list[i][0]
        v_val = V_list[i][1]
        print("g_key:", g_key)
        print("g_val_shape:", g_val.size())
        print("v_key:", v_key)
        print("v_val_shape:", v_val.size())
        g_val_np = g_val.numpy()
        v_val_np = v_val.numpy()
        weight_val = []
        for i in range(len(g_val_np)):
            g_val_l2 = g_val_np[i][0][0][0]
            v_val_l2 = np.linalg.norm(v_val_np[i])
            print("当前层g_val_模长：", g_val_l2)
            print("当前层v_val_模长：", v_val_l2)
            g_l2_div_v_l2 = g_val_l2 / v_val_l2
            print("|g|/|v|:", g_l2_div_v_l2)
            channel_weight = v_val_np[i] * g_l2_div_v_l2
            weight_val.append(channel_weight.tolist())
            print(np.array(weight_val).shape)
        weight_list.append(np.array(weight_val))

    print("weight_list_len:", len(weight_list))
        #logging.info('{}test'.format(a))a
    return weight_list


weight_list_x = get_weights(model_x)
weight_list_y = get_weights(model_y)


def get_std_list(model_name, weight_list):
    std_list = []
    mean_list = []
    max_list = []
    min_list = []
    for i in range(len(weight_list)):
        std = np.std(weight_list[i])
        logging.info('模型{}:第{}层卷积层参数的标准差为{:.3e}'.format(model_name, i + 1, std))
        std_list.append(std)
        mean = np.mean(weight_list[i])
        logging.info('模型{}:第{}层卷积层参数的均值为{:.3e}'.format(model_name, i + 1, mean))
        mean_list.append(mean)
        weight_max = np.max(weight_list[i])
        logging.info('模型{}:第{}层卷积层参数的最大值为{:.3e}'.format(model_name, i + 1, weight_max))
        max_list.append(weight_max)
        weight_min = np.min(weight_list[i])
        logging.info('模型{}:第{}层卷积层参数的最小值为{:.3e}'.format(model_name, i + 1, weight_min))
        min_list.append(weight_min)
    return std_list, mean_list, max_list, min_list


std_list_x, mean_list_x, max_list_x, min_list_x = get_std_list(model_x_name, weight_list_x)
std_list_y, mean_list_y, max_list_y, min_list_y = get_std_list(model_y_name, weight_list_y)


def statistics_save(path, mode, list_x, list_y, label_x, label_y):
    cnt = len(list_x)
    fig, ax = plt.subplots()
    ax.bar(np.arange(cnt), list_x, width=0.2, label=label_x)
    ax.bar(np.arange(cnt)+0.2, list_y, width=0.2, label=label_y)
    ax.set_xticks(np.arange(cnt)+0.1)
    ax.set_xticklabels(["第{}层".format(i + 1) for i in range(cnt)], fontsize='small')
    plt.title(mode, fontsize=20)
    plt.legend()
    plt.savefig(os.path.join(path, mode))
    return None


if not os.path.exists("results"):
    os.makedirs("results")
save_path = "results/" + model_x_name + "_with_" + model_y_name
if not os.path.exists(save_path):
    os.makedirs(save_path)


statistics_x = [std_list_x, mean_list_x, max_list_x, min_list_x]
statistics_y = [std_list_y, mean_list_y, max_list_y, min_list_y]
mode_list = ["std", "mean", "max", "min"]
for mode, (list_x, list_y) in zip(mode_list, zip(statistics_x, statistics_y)):
    statistics_save(save_path, mode, list_x, list_y, model_x_name, model_y_name)
