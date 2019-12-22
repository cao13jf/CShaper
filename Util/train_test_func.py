# -*- coding: utf-8 -*-

# import dependency library
from __future__ import absolute_import, print_function

# import user defined library
from Util.data_process import *


def volume_probability_prediction(temp_imgs, data_shape, label_shape, data_channel,
                                  class_num, batch_size, sess, proby, x):
    '''
    Test one image with sub regions along z-axis
    '''
    [D, H, W] = temp_imgs.shape
    input_center = [int(D/2), int(H/2), int(W/2)]
    temp_prob = np.zeros([D, H, W, class_num])
    sub_image_baches = []
    for center_slice in range(int(label_shape[0]/2), D + int(label_shape[0]/2), label_shape[0]):  # cover all center slice
        center_slice = min(center_slice, D - int(label_shape[0]/2))  # Make sure it can go up
        sub_image_bach = []
        temp_input_center = [center_slice, input_center[1], input_center[2]]
        sub_image = extract_roi_from_volume(temp_imgs, temp_input_center, data_shape)
        sub_image_bach.append(sub_image)
        sub_image_bach = np.asanyarray(sub_image_bach, np.float32)
        sub_image_baches.append(sub_image_bach)
        if (center_slice + label_shape[0]/2 > D):
            break
    total_batch = len(sub_image_baches)  # One image is devided into "total_batch"
    max_mini_batch = int((total_batch+batch_size-1)/batch_size)
    sub_label_idx = 0
    volume_full_flag = 0
    for mini_batch_idx in range(max_mini_batch):
        data_mini_batch = sub_image_baches[mini_batch_idx*batch_size:
                                      min((mini_batch_idx+1)*batch_size, total_batch)]  # Not exceed the maximum slice
        if(mini_batch_idx == max_mini_batch - 1):  # If some layers are left, then add some random noise slices.
            for idx in range(batch_size - (total_batch - mini_batch_idx*batch_size)):
                data_mini_batch.append(np.random.normal(0, 1, size = [data_channel] + data_shape))  #
        data_mini_batch = np.asarray(data_mini_batch, np.float32)
        data_mini_batch = np.transpose(data_mini_batch, [0, 2, 3, 4, 1])
        prob_mini_batch = sess.run(proby, feed_dict = {x: data_mini_batch})
        
        for batch_idx in range(prob_mini_batch.shape[0]):
            center_slice = sub_label_idx*label_shape[0] + int(label_shape[0]/2)

            if (center_slice >= D - int(label_shape[0]/2)):
                center_slice = D - int(label_shape[0]/2)
                volume_full_flag = 1
            temp_input_center = [center_slice, input_center[1], input_center[2], int(class_num/2)]

            sub_prob = np.reshape(prob_mini_batch[batch_idx], label_shape + [class_num])
            temp_prob = set_roi_to_volume(temp_prob, temp_input_center, sub_prob)   # Recover the volume from sub_pred
            sub_label_idx = sub_label_idx + 1
            if volume_full_flag:
                break
        # The batch center is determined by label_shape, and the number of sub_images is determined by the label_shape,
        # so the final pro after combine should be the same as the original image. 19 rawMemb slices are used to generate
        # only 11 probability slices.

    return temp_prob


def volume_probability_prediction_3d_roi(temp_imgs, data_shape, label_shape, data_channel, 
                                  class_num, batch_size, sess, proby, x):
    '''
    Test one image with sub regions along x, y, z axis  Sample on three directions
    '''
    [D, H, W] = temp_imgs[0].shape
    temp_prob = np.zeros([D, H, W, class_num])
    sub_image_batches = []
    sub_image_centers = []
    roid_half = int(label_shape[0]/2)
    roih_half = int(label_shape[1]/2)
    roiw_half = int(label_shape[2]/2)
    for centerd in range(roid_half, D + roid_half, label_shape[0]):
        centerd = min(centerd, D - roid_half)
        for centerh in range(roih_half, H + roih_half, label_shape[1]):
            centerh =  min(centerh, H - roih_half) 
            for centerw in range(roiw_half, W + roiw_half, label_shape[2]):
                centerw =  min(centerw, W - roiw_half) 
                temp_input_center = [centerd, centerh, centerw]
                sub_image_centers.append(temp_input_center)
                sub_image_batch = []
                for chn in range(data_channel):
                    sub_image = extract_roi_from_volume(temp_imgs[chn], temp_input_center, data_shape)
                    sub_image_batch.append(sub_image)
                sub_image_bach = np.asanyarray(sub_image_batch, np.float32)
                sub_image_batches.append(sub_image_bach)

    total_batch = len(sub_image_batches)
    max_mini_batch = int((total_batch + batch_size - 1)/batch_size)
    sub_label_idx = 0
    for mini_batch_idx in range(max_mini_batch):
        data_mini_batch = sub_image_batches[mini_batch_idx*batch_size:
                                      min((mini_batch_idx+1)*batch_size, total_batch)]
        if(mini_batch_idx == max_mini_batch - 1):
            for idx in range(batch_size - (total_batch - mini_batch_idx*batch_size)):
                data_mini_batch.append(np.random.normal(0, 1, size = [data_channel] + data_shape))
        data_mini_batch = np.asanyarray(data_mini_batch, np.float32)
        data_mini_batch = np.transpose(data_mini_batch, [0, 2, 3, 4, 1])
        outprob_mini_batch = sess.run(proby, feed_dict = {x: data_mini_batch})
        
        for batch_idx in range(batch_size):
            glb_batch_idx = batch_idx + mini_batch_idx * batch_size
            if(glb_batch_idx >= total_batch):
                continue
            temp_center = sub_image_centers[glb_batch_idx]
            temp_prob = set_roi_to_volume(temp_prob, temp_center + [1], outprob_mini_batch[batch_idx])
            sub_label_idx = sub_label_idx + 1

    return temp_prob


def volume_probability_prediction_dynamic_shape(temp_imgs, data_shape, label_shape, data_channel, 
                                  class_num, batch_size, sess, proby, x):
    '''
    Test one image with sub regions along z-axis
    The height and width of input tensor is adapted to those of the input image
    '''
    # construct graph
    [D, H, W] = temp_imgs.shape
    Hx = max(int((H+3)/4)*4, data_shape[1])  # TODO: why do this
    Wx = max(int((W+3)/4)*4, data_shape[2])
    data_slice = data_shape[0]
    label_slice = label_shape[0]
    # full_data_shape = [batch_size, data_slice, Hx, Wx, data_channel]
    # x = tf.placeholder(tf.float32, full_data_shape)
    
    new_data_shape = [data_slice, Hx, Wx]
    new_label_shape = [label_slice, Hx, Wx]
    temp_prob = volume_probability_prediction(temp_imgs, new_data_shape, new_label_shape, data_channel, 
                                              class_num, batch_size, sess, proby, x)
    return temp_prob


def prediction_fusion(prob_sagittal, prob_axial):
    '''
    Deal with combining prediction from different directions
    :param pred_list:
    :return:
    '''

    # Determine which part of axial should be fused with sagittal
    largest = np.amax(prob_sagittal, axis=-1)
    ## Use shift-max average to ignore intrinsical uncertainty on boundary
    largest_translate = ndimage.shift(largest, [1, 1, 1], mode='constant', cval=0)
    smoothed_largest = np.maximum(largest, largest_translate)
    bin_prob0 = smoothed_largest < 0.5
    [idx, idy, idz] = np.nonzero(bin_prob0)

    # Hard fusion embedded with segMemb
    pred_axial = np.argmax(prob_axial, axis=-1).astype(np.uint16)
    pred_sagittal = np.argmax(prob_sagittal, axis=-1).astype(np.int16)
    pred_fused = np.copy(pred_sagittal)
    pred_fused[idx, idy, idz] = pred_axial[idx, idy, idz]  # TODO: this step is too time-consuming

    return pred_fused


def test_one_image_three_nets_adaptive_shape(temp_imgs, data_shapes, label_shapes, data_channel, class_num,
                   batch_size, sess, nets, outputs, inputs, proby, x, shape_mode):
    '''
    Test one image with three anisotropic networks with fixed or adaptable tensor height and width.
    These networks are used in axial, saggital and coronal view respectively.
    shape_mode: 0: use fixed tensor shape in all direction
                1: compare tensor shape and image shape and then select fixed or adaptive tensor shape
                2: use adaptive tensor shape in all direction
    '''

    [ax_data_shape] = data_shapes  # Slice volume in three different directions, and then average them
    [ax_label_shape] = label_shapes
    [D, H, W] = temp_imgs.shape  # Shape is defined based on the shape relationship between given and fixed
    if(shape_mode == 0 or (shape_mode == 1 and (H <= ax_data_shape[1] and W <= ax_data_shape[2]))):
        prob = volume_probability_prediction(temp_imgs, ax_data_shape, ax_label_shape, data_channel,
                                  class_num, batch_size, sess, outputs[0], inputs[0])
    else:
        prob = volume_probability_prediction_dynamic_shape(temp_imgs, ax_data_shape, ax_label_shape, data_channel,
                                  class_num, batch_size, sess, proby, x)

    return prob
