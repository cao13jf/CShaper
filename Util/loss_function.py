# -*- coding: utf-8 -*-
#
#
# from niftynet.layer.loss_segmentation import cross_entropy
import tensorflow as tf

def weighted_one_hot_loss(output_channels):
    def get_weighted_one_hot_loss(ylabel, pred0):
        pred = tf.reshape(pred0, (-1, output_channels))
        epsilon = tf.constant(value=1e-25)
        predSoftmax = tf.cast(tf.nn.softmax(pred), tf.float32)

        gt = tf.one_hot(indices=tf.cast(tf.squeeze(tf.reshape(ylabel, (-1, 1))), tf.int32), depth=output_channels, dtype=tf.float32)

        ## Construct weight matrix for loss function
        ylabel_flat = tf.cast(tf.reshape(ylabel, (-1, 1)), tf.float32)
        zero_dis = tf.zeros_like(ylabel_flat)
        label_dis = tf.zeros_like(ylabel_flat)
        for i in range(1, output_channels):
            label_dis = tf.concat([label_dis, tf.abs(zero_dis + i - ylabel_flat)], axis=1)
        weight = tf.exp(label_dis/tf.reduce_max(label_dis))
        #weight = tf.to_float(tf.reshape(weight, (-1, 1)))

        crossEntropyScaling = tf.cast([0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5], tf.float32)
        #crossEntropyScaling = tf.to_float([0.9, 1.0])
        crossEntropy = -tf.compat.v1.reduce_sum(((1 - gt) * tf.math.log(tf.maximum(1 - predSoftmax, epsilon))
                                       + gt * tf.math.log(tf.maximum(predSoftmax, epsilon))) * weight * crossEntropyScaling,
                                      reduction_indices=[1])  # Final result should be in a row

        crossEntropySum = tf.compat.v1.reduce_sum(crossEntropy, name="cross_entropy_sum")

        return crossEntropySum

    return get_weighted_one_hot_loss

def regression_loss(predicty, ylabel, weight_map=None):
    """
    Distance regression loss
    :param predicty: prediction results
    :param ylabel: ground truth
    :return: regression loss
    """
    loss = tf.nn.l2_loss(predicty, ylabel, weight_map)

    return loss