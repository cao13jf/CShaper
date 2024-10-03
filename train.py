# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function

#  import dependency library
import os
import argparse
import setproctitle
import time
import tensorflow as tf
from random import random
from shutil import copyfile
from tensorflow.keras import regularizers

#  import user defined library
from Util.loss_function import weighted_one_hot_loss
from Util.data_loader import DataGene
from Util.parse_config import parse_config
from Util.DMapNetUpdated import DMapNetCompiled
tf.compat.v1.disable_eager_execution()
os.environ['KMP_DUPLICATE_LIB_OK']='True'

#=============================================================
#          main function for training
#=============================================================
def train(config_file, train_ratio=1.0):

    setproctitle.setproctitle("train_ratio:" + str(train_ratio))
    #=============================================================
    #               1, Load configuration parameters
    #=============================================================
    config = parse_config(config_file)
    config_data = config['data']
    config_data["train_ratio"] = train_ratio
    config_net = config['network']
    config_train = config['training']
    # random.seed(config_train.get('random_seed', 1))
    assert(config_data['with_ground_truth'])
    class_num = config_data['edt_discrete_num']


    #==============================================================
    #               2, Construct computation graph
    #==============================================================
    tf.compat.v1.reset_default_graph()
    with tf.name_scope('model_builder'):
        w_regularizer = regularizers.L2(config_train.get('decay', 1e-7))
        b_regularizer = regularizers.L2(config_train.get('decay', 1e-7))
        net = DMapNetCompiled(input_size=config_data['data_shape'],
                              num_classes=class_num,
                              kernel_regularizer=w_regularizer,
                              bias_regularizer=b_regularizer,
                              activation="relu")

    #==============================================================
    #               3, Data loader
    #==============================================================
    dataloader = DataGene(config_data)
    epoches = config_train["maximal_iteration"] // len(dataloader.data)
    opt = tf.keras.optimizers.Adam(learning_rate=config_train["learning_rate"])
    net.compile(optimizer=opt, loss=weighted_one_hot_loss(config_data['edt_discrete_num']))
    results = net.fit(dataloader, epochs=epoches, shuffle=True, workers=8)

    # ==============================================================
    #               3, Start train
    # ==============================================================
    net.save_weights(os.path.join(config_train["summary_dir"], config_train['model_save_prefix'] + "_{}.ckpt".format(config_train['maximal_iteration'])))
    timestr = time.strftime('%m_%d_%H_%M')
    copyfile(config_file, os.path.join(os.path.dirname(config_train['model_save_prefix']), config_net['net_name']+'Paramters-'+timestr+'.txt'))

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--cf", required=True)
    args.add_argument("--train_ratio", type=float, default=1)
    args = args.parse_args()
    assert (os.path.isfile(args.cf)), "Config file {} doesn't ecist".format(args.cf)  # make sure config_file is a file name
    assert (args.train_ratio > 0) & (args.train_ratio < 1.000001), "Invalid train ratio (0 < x < 1): {}".format(args.train_ratio)
    train(config_file=args.cf, train_ratio=args.train_ratio)
