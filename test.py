from __future__ import absolute_import, print_function

# import dependency library

# import user defined library
import os
import time
import argparse
import shutil
import numpy as np
from tqdm import tqdm
from util.data_process import transpose_volumes, delete_isolate_labels, post_process_on_edt, set_crop_to_volume, transpose_volumes_reverse, save_array_as_nifty_volume
from dataset.data_loader import DataGene

from util.train_test_func import predict_full_volume
from util.segmentation_post_process import post_process
from util.parse_config import parse_config
from util.train_test_func import prediction_fusion
from model.DMapNetUpdated import DMapNetCompiled
import gc


def test(config):
    # group parameters
    config = parse_config(config)
    config_data = config['data']
    config_net = config.get('network', None)
    config_test = config['testing']
    batch_size = config_test.get('batch_size', 5)
    label_edt_discrete = config_data.get('label_edt_discrete', False)
    if not config_test['only_post_process']:
        data_shape = config_net['data_shape']
        label_shape = config_net['label_shape']
        class_num = config_data.get('edt_discrete_num', 16)

        # ==============================================================
        #               3, Data loader
        # ==============================================================
        dataloader = DataGene(config_data)
        [temp_imgs, img_names, emrbyo_name, temp_bbox, temp_size] = dataloader.get_image_data_with_name(0)

        temp_img_axial = transpose_volumes(temp_imgs, slice_direction='axial')
        [D, H, W] = temp_img_axial.shape
        Hx = max(int((H + 3) / 4) * 4, data_shape[1])
        Wx = max(int((W + 3) / 4) * 4, data_shape[2])
        data_slice = data_shape[0]
        full_data_shape = [data_slice, Hx, Wx, data_shape[-1]]  # [None, data_slice, Hx, Wx, data_shape[-1]]
        net_axial = DMapNetCompiled(input_size=full_data_shape,
                                    num_classes=config_data['edt_discrete_num'])
        net_axial.load_weights(config_test["model_file"])

        temp_imgs = transpose_volumes(temp_imgs, slice_direction='sagittal')
        [D, H, W] = temp_imgs.shape
        Hx = max(int((H + 3) / 4) * 4, data_shape[1])
        Wx = max(int((W + 3) / 4) * 4, data_shape[2])
        data_slice = data_shape[0]
        full_data_shape = [data_slice, Hx, Wx, data_shape[-1]]
        net_sagittal = DMapNetCompiled(input_size=full_data_shape,
                                       num_classes=config_data['edt_discrete_num'],
                                       activation="relu")
        net_sagittal.load_weights(config_test["model_file"])

        # ==============================================================
        #               4, Start prediction
        # ==============================================================
        slice_direction = config_test.get('slice_direction', 'axial')
        save_folder = config_data['save_folder']
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)  # If the target folder doesn't exist, create a new one
        test_time = []
        data_number = config_data.get('max_time', 100) * len(config_data["data_names"])
        if os.path.isdir(os.path.join(save_folder, emrbyo_name)):
            shutil.rmtree(os.path.join(save_folder, emrbyo_name))
        for i in tqdm(range(0, data_number), desc='Extracting binary membrane'):
            [temp_imgs, img_names, emrbyo_name, temp_bbox, temp_size] = dataloader.get_image_data_with_name(i)
            t0 = time.time()

            temp_img_sagittal = transpose_volumes(temp_imgs, slice_direction="sagittal")
            prob_sagittal = predict_full_volume(temp_img_sagittal, data_shape[:-1], label_shape[:-1], data_shape[-1],
                                                class_num, batch_size, net_sagittal)
            temp_img_axial = transpose_volumes(temp_imgs, slice_direction='axial')
            prob_axial = predict_full_volume(temp_img_axial, data_shape[:-1], label_shape[:-1], data_shape[-1],
                                             class_num, batch_size, net_axial)

            # If the prediction is one-hot tensor, use np.argmax to get the indices of the maximumm. That indice is used as label
            if(label_edt_discrete):  # If we hope to predict discrete EDT map, argmax on the indices depth
                if config_test.get('direction_fusion', False):
                    prob_axial = np.transpose(prob_axial, [2, 0, 1, 3])
                    pred_fusion = prediction_fusion(prob_axial, prob_sagittal)
                else:
                    pred_fusion = (np.argmax(prob_sagittal, axis=-1)).astype(np.uint16)
                pred = delete_isolate_labels(pred_fusion)
            else:
                pred = prob_sagittal  # Regression prediction provides the MAP directly.

            out_label = post_process_on_edt(pred).astype(np.int16)

            # ==============================================================
            #               save prediction results as *.nii.gz
            # ==============================================================
            test_time.append(time.time() - t0)
            final_label = np.zeros(temp_size, out_label.dtype)
            out_label = np.transpose(out_label, [1, 2, 0])
            final_label = set_crop_to_volume(final_label, temp_bbox[0], temp_bbox[1], out_label)
            final_label = transpose_volumes_reverse(final_label, slice_direction)
            save_file = os.path.join(save_folder, emrbyo_name, img_names.replace(".nii.gz", "_segMemb.nii.gz"))
            save_array_as_nifty_volume(final_label, save_file) # os.path.join(save_folder, one_embryo, one_embryo + "_" + tp_str.zfill(3) + "_cell.nii.gz")



        del dataloader, net_sagittal, net_axial
        if __name__ != '__main__':
            gc.collect()


    # ==============================================================
    #               Post processing (binary membrane --> isolated SegCell)
    # ==============================================================
    if config_test.get('post_process', False):
        config_post = {}
        config_post['segdata'] = config['segdata']
        config_post['segdata']['embryos'] = config_data['data_names']
        config_post['result'] = {}

        config_post['segdata']['membseg_path'] = config_data['save_folder']
        config_post['result']['postseg_folder'] = config_data['save_folder'] + 'Postseg'
        config_post['result']['nucleus_filter'] = config_test['nucleus_filter']
        config_post['result']['nucleus_as_seed'] = config_test['nucleus_as_seed']
        config_post['segdata']['max_time'] = config["data"]["max_time"]
        if not os.path.isdir(config_post['result']['postseg_folder']):
            os.makedirs(config_post['result']['postseg_folder'])

        post_process(config_post)


if __name__ == '__main__':
    st = time.time()
    args = argparse.ArgumentParser()
    args.add_argument("--cf", required=True)
    args = args.parse_args()
    assert (os.path.isfile(args.cf)), "Configure file {} doesn't exist".format(args.cf)
    test(args.cf)  # < ----- input parameters here

