# import denpendency library
import os
import glob
import pickle
import torch
import random
import numpy as np
import shutil


M = 2**32 -1

def check_folder(file_folder, overwrite=False):
    if "." in os.path.basename(file_folder):
        file_folder = os.path.dirname(file_folder)
    if os.path.isdir(file_folder) and overwrite:
        shutil.rmtree(file_folder)
    elif not os.path.isdir(file_folder):
        os.makedirs(file_folder)

def get_all_stack(root, membrane_list, suffix, max_times, sub_dir="RawMemb"):
    file_list = []
    if isinstance(root, list):
        root = root[0]
    for idx, membrane in enumerate(membrane_list):
        max_time = max_times or -1
        stacks = glob.glob(os.path.join(root, membrane, sub_dir, suffix))
        stacks = sorted(stacks)[:max_time]
        file_list = file_list + stacks
    return file_list

def pkload(fname):
    with open(fname, "rb") as f:
        return pickle.load(f)

#  initilization function for workers
def init_fn(worker):
    seed = torch.LongTensor(1).random_().item()
    seed = (seed + worker) % M
    np.random.seed(seed)
    random.seed(seed)

def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):  #
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power( 1 - (epoch) / MAX_EPOCHES ,power),8)

def mse_loss(output, target, *args):
    weights = (0.2 * (target - target.min()) + (target - target.min()).mean())
    loss = 0.5 * (weights * (target - output) ** 2).mean()

    return loss
