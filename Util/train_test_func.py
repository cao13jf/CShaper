# -*- coding: utf-8 -*-

# import dependency library
from __future__ import absolute_import, print_function
import numpy as np
from scipy import ndimage

# import user defined library
from Util.data_process import crop_from_volume, merge_crop_to_volume

def predict_full_volume(img, data_shape, label_shape, data_channel, class_num,batch_size, net):
    '''
    Predict on one image.
    :Param img: full volume raw image
    :data_shape：slice data shape
    :label_shape: slice label shape
    :data_channel: number of data channels
    :class_num: number of classes
    :batch_size: batch size
    :sess: tf session
    :proby: prediction
    :x: input
    '''
    [D, H, W] = img.shape
    Hx = max(int((H + 3) / 4) * 4, data_shape[1])
    Wx = max(int((W + 3) / 4) * 4, data_shape[2])
    data_slice = data_shape[0]
    label_slice = label_shape[0]

    new_data_shape = [data_slice, Hx, Wx]
    new_label_shape = [label_slice, Hx, Wx]
    temp_prob = volume_prediction(img, new_data_shape, new_label_shape, data_channel, class_num, batch_size, net)
    return temp_prob


def volume_prediction(img, data_shape, label_shape, data_channel, class_num, batch_size, net):
    '''
    Test one image with sub regions along z-axis
    '''
    [D, H, W] = img.shape
    input_center = [int(D/2), int(H/2), int(W/2)]
    temp_prob = np.zeros([D, H, W, class_num])
    sub_image_baches = []
    for center_slice in range(int(label_shape[0]/2), D + int(label_shape[0]/2), label_shape[0]):  # cover all center slice
        center_slice = min(center_slice, D - int(label_shape[0]/2))  # Make sure it can go up
        sub_image_bach = []
        temp_input_center = [center_slice, input_center[1], input_center[2]]
        sub_image = crop_from_volume(img, temp_input_center, data_shape)
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
                data_mini_batch.append(np.random.normal(0, 1, size=[data_channel] + data_shape))  #
        data_mini_batch = np.asarray(data_mini_batch, np.float32)
        data_mini_batch = np.transpose(data_mini_batch, [0, 2, 3, 4, 1])
        prob_mini_batch = net.predict(data_mini_batch)
        
        for batch_idx in range(prob_mini_batch.shape[0]):
            center_slice = sub_label_idx*label_shape[0] + int(label_shape[0]/2)

            if (center_slice >= D - int(label_shape[0]/2)):
                center_slice = D - int(label_shape[0]/2)
                volume_full_flag = 1
            temp_input_center = [center_slice, input_center[1], input_center[2], int(class_num/2)]

            sub_prob = np.reshape(prob_mini_batch[batch_idx], label_shape + [class_num])
            temp_prob = merge_crop_to_volume(temp_prob, temp_input_center, sub_prob)   # Recover the volume from sub_pred
            sub_label_idx = sub_label_idx + 1
            if volume_full_flag:
                break
        # The batch center is determined by label_shape, and the number of sub_images is determined by the label_shape,
        # so the final pro after combine should be the same as the original image. 19 RawMemb slices are used to generate
        # only 11 probability slices.

    return temp_prob

def prediction_fusion(prob_sagittal, prob_axial):
    '''
    Deal with combining prediction from different directions
    :param pred_list:
    :return fused results on different directions:
    '''

    # Determine which part of axial should be fused with sagittal
    largest = np.amax(prob_sagittal, axis=-1)
    ## Use shift-max average to ignore intrinsical uncertainty on boundary
    largest_translate = ndimage.shift(largest, [1, 1, 1], mode='constant', cval=0)
    smoothed_largest = np.maximum(largest, largest_translate)
    bin_prob0 = smoothed_largest < 0.5
    [idx, idy, idz] = np.nonzero(bin_prob0)

    # Hard fusion embedded with SegMemb
    pred_axial = np.argmax(prob_axial, axis=-1).astype(np.uint16)
    pred_sagittal = np.argmax(prob_sagittal, axis=-1).astype(np.int16)
    pred_fused = np.copy(pred_sagittal)
    pred_fused[idx, idy, idz] = pred_axial[idx, idy, idz]

    return pred_fused