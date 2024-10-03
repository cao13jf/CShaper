#  import dependency library
import os
import argparse
import logging
import random
import shutil
import numpy as np
import torch
import setproctitle
import torch.optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

#  import user's library
from Util.DMFNet16 import EDTDMFNet

from Util.torch_dataset import Memb3DDataset

from utils.prediction_utils import validate, membrane2cell, combine_dividing_cells
from utils.shape_analysis import shape_analysis_func
from utils.qc import generate_qc
from utils.generate_gui_data import generate_gui_data

cudnn.benchmark = True # https://zhuanlan.zhihu.com/p/73711222 to accelerate the network

# =========================================================
#  main program for prediction
# =========================================================
def test(config_test):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    assert torch.cuda.is_available(), "CPU is needed for prediction"


    # get membrane binary shell
    test_folder = dict(root="dataset/test", has_label=False)
    # =============================================================
    #  construct network model
    # =============================================================
    model = EDTDMFNet(in_channels=1, n_first=32, conv_channels=64, groups=16, norm="in", out_class=1)
    check_point = torch.load(config_test["model_file"])
    model.load_state_dict(check_point["state_dict"])

    data_root = config_test['data_root'] if type(config_test['data_root']) is list else [config_test['data_root']]  # Save as list
    data_names = config_test.get('data_names', None)
    augmentations = "Compose([ContourEDT(9),RandomIntensityChange([0.1, 0.1]),RandCrop((128,128,128)),RandomFlip(0),NumpyType((np.float32, np.float32, np.float32, np.float32))])"
    # =============================================================
    #    set data loader
    # =============================================================
    test_set = Memb3DDataset(root=data_root, membrane_names=data_names, for_train=False, transforms=augmentations,
                       return_target=False, suffix="*.nii.gz", max_times=config_test["max_time"])
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=1,
        shuffle=False,
        # collate_fn=test_set.collate, # control how data is stacked
        num_workers=10,
        pin_memory=True
    )

    #=============================================================
    #  begin prediction
    #=============================================================
    #  Prepare (or clear) in order to update all files
    save_folder = config_test['save_folder']
    for embryo_name in data_names:
        if os.path.isdir(os.path.join(save_folder, embryo_name)):
            shutil.rmtree(os.path.join(save_folder, embryo_name))

    # the file will save in the segMemb folder
    with torch.no_grad():
        validate(
            valid_loader=test_loader,  # dataset loader
            model=model,  # model
            savepath= save_folder,  # output folder
            names=test_set.names,  # stack name lists
            scoring=False,  # whether keep accuracy
            save_format=".nii.gz",  # save volume format
            snapsot=visualizer,  # whether keep snap
            postprocess=False,
            size=test_set.size
        )


    membrane2cell(config_test)

    combine_dividing_cells(config_test)

if __name__ == "__main__":
    test()