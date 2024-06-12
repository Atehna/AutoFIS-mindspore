# 运行AutoFIS 应用于DeepFM调用init,trainer,models
# encoding=utf-8
import sys
import time
import os
import __init__



sys.path.append(__init__.config['data_path'])  # add your data path here
from datasets import as_dataset
from ms_trainer import Trainer
from ms_models import AutoDeepFM
from ms_utils import get_optimizer, get_loss  # 确保导入get_loss函数
import mindspore as ms
import traceback

seeds = [0x0123, 0x4567, 0x3210, 0x7654, 0x89AB, 0xCDEF, 0xBA98, 0xFEDC,
         0x0123, 0x4567, 0x3210, 0x7654, 0x89AB, 0xCDEF, 0xBA98, 0xFEDC]
data_name = 'avazu'
dataset = as_dataset(data_name)
backend = 'ms'
batch_size = 2000

train_data_param = {
    'gen_type': 'train',
    'random_sample': True,
    'batch_size': batch_size,
    'split_fields': False,
    'on_disk': True,
    'squeeze_output': True,
}
test_data_param = {
    'gen_type': 'test',
    'random_sample': False,
    'batch_size': batch_size,
    'split_fields': False,
    'on_disk': True,
    'squeeze_output': True,
}


def run_one_model(model=None, learning_rate=1e-3, decay_rate=1.0, epsilon=1e-8, ep=5, grda_c=0.005,
                  grda_mu=0.51, learning_rate2=1e-3, decay_rate2=1.0, retrain_stage=0):
    n_ep = ep * 1
    train_param = {
        'opt1': 'adam',
        'opt2': 'grda',
        'pos_weight': 1.0,
        'n_epoch': n_ep,
        'train_per_epoch': dataset.train_size / ep,  # split training data
        'test_per_epoch': dataset.test_size,
        'early_stop_epoch': int(0.5 * ep),
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'decay_rate': decay_rate,
        'learning_rate2': learning_rate2,
        'decay_rate2': decay_rate2,
        'epsilon': epsilon,
        'load_ckpt': False,
        'ckpt_time': 10000,
        'grda_c': grda_c,
        'grda_mu': grda_mu,
        'test_every_epoch': int(ep / 5),
        'retrain_stage': retrain_stage,
    }

    # 创建数据生成器【训练集和测试集】，train_data_param配置参数，生成批量的测试或者训练数据
    train_gen = dataset.batch_generator(train_data_param)
    test_gen = dataset.batch_generator(test_data_param)

    # 获取损失函数
    loss_fn = get_loss('weight')(pos_weight=train_param['pos_weight'])  # 创建带权重的损失函数实例

    # 创建训练对象器，初始化 Trainer 对象，传入模型 model、训练数据生成器 train_gen 和测试数据生成器 test_gen，以及其他训练参数 train_param。
    trainer = Trainer(model=model, train_gen=train_gen, test_gen=test_gen, loss_fn=loss_fn, **train_param)

    # 开始训练
    trainer.fit()


import math

if __name__ == "__main__":
    # general parameter
    embedding_size = 40
    l2_v = 0.0
    depth = 5
    width = 700
    ls = [width] * depth
    ls.append(1)
    la = ['relu'] * depth
    la.append(None)
    lk = [1.0] * depth
    lk.append(1.0)
    batch_norm = True
    layer_norm = False
    learning_rate = 1e-3
    dc = 0.7
    split_epoch = 5

    # second-order parameter
    weight_base = 0.6  # the initial value of alpha
    comb_mask = None  # which interactions are reserved

    # third-order parameter
    third_prune = False  # whether condisder third-order feature interaction
    weight_base_third = 0.6
    comb_mask_third = None

    # search_stage or retrain_stage; 0 represents search stage and 1 represents retrain stage
    retrain_stage = 0  # in retrain stage, optimize all parameters by adam Optimizer, you need to mask interactions by comb_mask and comb_mask_third

    # grda parameter
    grda_c = 0.0005
    grda_mu = 0.8
    learning_rate2 = 1.0  # learning rate for alpha in research stage
    dc2 = 1.0
    model = AutoDeepFM(init="xavier", num_inputs=dataset.max_length, input_dim=dataset.num_features,
                       l2_v=l2_v, layer_sizes=ls, layer_acts=la, layer_keeps=lk, layer_l2=[0, 0],
                       embed_size=embedding_size, batch_norm=batch_norm, layer_norm=layer_norm,
                       comb_mask=comb_mask, weight_base=weight_base, third_prune=third_prune,
                       weight_base_third=weight_base_third, comb_mask_third=comb_mask_third,
                       retrain_stage=retrain_stage)
    run_one_model(model=model, learning_rate=learning_rate, epsilon=1e-8,
                  decay_rate=dc, ep=split_epoch, grda_c=grda_c, grda_mu=grda_mu,
                  learning_rate2=learning_rate2, decay_rate2=dc2, retrain_stage=retrain_stage)
