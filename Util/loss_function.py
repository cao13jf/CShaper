# -*- coding: utf-8 -*-
#
#


from niftynet.layer.loss_segmentation import cross_entropy
from niftynet.layer.loss_regression import l2_loss
import tensorflow as tf
import numpy as np


def one_hot_loss(predicty, ylabel, class_num, weight_map=None):
    """
    one hot loss function
    :param predicty: prediction
    :param ylabel: ground truth
    :param class_num: number of classes in one hot
    :param weight_map: weight for loss function
    :return: one hot cross entropy loss function
    """
    yshape = ylabel.shape
    ylabel  = tf.squeeze(ylabel)
    one_hot_label = tf.one_hot(ylabel, depth=class_num)
    loss = cross_entropy(predicty, one_hot_label, weight_map)

    # For recording
    sample_slice = ylabel[0, 0,:,:]
    pred_out  =  tf.argmax(predicty, axis=-1)
    print(pred_out.shape)
    pred_slice   = pred_out[0, 0,:,:]
    sample_slice = tf.cast(sample_slice, tf.float32)/16
    pred_slice = tf.cast(pred_slice, tf.float32)/16
    sample_slice = tf.reshape(sample_slice, [1, yshape[2], yshape[3], 1])
    pred_slice = tf.reshape(pred_slice, [1, yshape[2], yshape[3], 1])
    tf.summary.image('Input', sample_slice, 3)
    tf.summary.image('Output', pred_slice, 3)

    return loss

def weighted_one_hot_loss(pred0, ylabel, output_channels, weight=None, ss=None):
    pred = tf.reshape(pred0, (-1, output_channels))
    epsilon = tf.constant(value=1e-25)
    predSoftmax = tf.to_float(tf.nn.softmax(pred))

    gt = tf.one_hot(indices=tf.to_int32(tf.squeeze(tf.reshape(ylabel, (-1, 1)))), depth=output_channels, dtype=tf.float32)

    ## Construct weight matrix for loss function
    ylabel_flat = tf.to_float(tf.reshape(ylabel, (-1, 1)))
    zero_dis = tf.zeros_like(ylabel_flat)
    label_dis = tf.zeros_like(ylabel_flat)
    for i in range(1, output_channels):
        label_dis = tf.concat([label_dis, tf.abs(zero_dis + i - ylabel_flat)], axis=1)
    weight = tf.exp(label_dis/tf.reduce_max(label_dis))
    #weight = tf.to_float(tf.reshape(weight, (-1, 1)))

    crossEntropyScaling = tf.to_float([0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
    #crossEntropyScaling = tf.to_float([0.9, 1.0])
    # TODO: loss may also be weighted by "classes distance", 1<->3 is closer then 1<->10.
    crossEntropy = -tf.reduce_sum(((1 - gt) * tf.log(tf.maximum(1 - predSoftmax, epsilon))
                                   + gt * tf.log(tf.maximum(predSoftmax, epsilon))) * weight * crossEntropyScaling,
                                  reduction_indices=[1])  # Final result should be in a row

    crossEntropySum = tf.reduce_sum(crossEntropy, name="cross_entropy_sum")

    return crossEntropySum

def regression_loss(predicty, ylabel, weight_map=None):
    """
    Distance regression loss
    :param predicty: prediction results
    :param ylabel: ground truth
    :return: regression loss
    """
    loss = l2_loss(predicty, ylabel, weight_map)

    return loss