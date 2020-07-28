from __future__ import absolute_import, print_function

# import dependency library
import tensorflow as tf

# import user defined library
from Util.data_loader import *
from Util.train_test_func import *
from Util.segmentation_post_process import *
from Util.parse_config import parse_config
from Util.train_test_func import prediction_fusion
from train import NetFactory

def test(config_file):

    #=============================================================
    #               1, Load configuration parameters
    #=============================================================
    config = parse_config(config_file)
    config_data = config['data']
    config_net = config.get('network', None)
    config_test = config['testing']
    batch_size = config_test.get('batch_size', 5)
    label_edt_discrete = config_data.get('label_edt_discrete', False)
    if not config_test['only_post_process']:
        net_type = config_net['net_type']
        net_name = config_net['net_name']
        data_shape = config_net['data_shape']
        label_shape = config_net['label_shape']
        class_num = config_data.get('edt_discrete_num', 16)


        # ==============================================================
        #               2, Construct computation graph
        # ==============================================================
        full_data_shape = [batch_size] + data_shape
        input_x = tf.placeholder(tf.float32, shape=full_data_shape)
        net_class = NetFactory.create(net_type)
        net = net_class(num_classes=class_num, w_regularizer=None, b_regularizer=None, name=net_name)
        net.set_params(config_net)

        net_output = net(input_x, is_training=True)
        if (label_edt_discrete):
            net_result = tf.nn.softmax(net_output)
        else:
            net_result = net_output

        # ==============================================================
        #               3, Data loader
        # ==============================================================
        dataloader = DataLoader(config_data)
        dataloader.load_data()
        [temp_imgs, temp_weight, img_names, emrbyo_name, temp_bbox, temp_size] = dataloader.get_image_data_with_name(0)
        
        # For axial direction
        temp_img_axial = transpose_volumes(temp_imgs, slice_direction='axial')
        [D, H, W] = temp_img_axial.shape
        Hx = max(int((H + 3) / 4) * 4, data_shape[1])  
        Wx = max(int((W + 3) / 4) * 4, data_shape[2])
        data_slice = data_shape[0]
        full_data_shape = [batch_size, data_slice, Hx, Wx, data_shape[-1]]
        x_axial = tf.placeholder(tf.float32, full_data_shape)
        predicty_axial = net(x_axial, is_training=True)
        proby_axial = tf.nn.softmax(predicty_axial)

        # For sagittal direction
        temp_imgs = transpose_volumes(temp_imgs, slice_direction='sagittal')
        [D, H, W] = temp_imgs.shape
        Hx = max(int((H + 3) / 4) * 4, data_shape[1])  
        Wx = max(int((W + 3) / 4) * 4, data_shape[2])
        data_slice = data_shape[0]
        full_data_shape = [batch_size, data_slice, Hx, Wx, data_shape[-1]]
        x_sagittal = tf.placeholder(tf.float32, full_data_shape)
        predicty_sagittal = net(x_sagittal, is_training=True)
        proby_sagittal = tf.nn.softmax(predicty_sagittal)

        # ==============================================================
        #               4, Start prediction
        # ==============================================================
        all_vars = tf.global_variables()
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        net1_vars = [x for x in all_vars if x.name[0:len(net_name) + 1] == net_name + '/']
        saver = tf.train.Saver(net1_vars)
        saver.restore(sess, config_net['model_file'])
        sess.graph.finalize()
        slice_direction = config_test.get('slice_direction', 'axial')
        save_folder = config_data['save_folder']
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)  # If the target folder doesn't exist, create a new one
        test_time = []
        data_number = config_data.get('max_time', 100) * len(config_data["data_names"])
        for i in tqdm(range(0, data_number), desc='Segmenting Data:'):
            [temp_imgs, img_names, emrbyo_name, temp_bbox, temp_size] = dataloader.get_image_data_with_name(i)
            temp_img_sagittal = transpose_volumes(temp_imgs, slice_direction)
            if (slice_direction == 'sagittal'):
                tem_box = temp_bbox.copy()
                temp_bbox = [[a[2], a[0], a[1]] for a in tem_box]
                temp_size = (temp_size[2], temp_size[0], temp_size[1])
            if (slice_direction == 'coronal'):
                tem_box = temp_bbox.copy()
                temp_bbox = [[a[1], a[0], a[2]] for a in tem_box]
                temp_size = (temp_size[1], temp_size[0], temp_size[2])
            t0 = time.time()
            prob_sagittal = predict_full_volume(temp_img_sagittal, data_shape[:-1], label_shape[:-1], data_shape[-1],
                                                class_num, batch_size, sess, proby_sagittal, x_sagittal)
            # Combine results from two different directions. In fusion stage, refinement is based on sagittal direction
            if config_test.get('direction_fusion', False):
                temp_img_axial = transpose_volumes(temp_imgs, slice_direction = 'axial')
                prob_axial = predict_full_volume(temp_img_axial, data_shape[:-1], label_shape[:-1], data_shape[-1],
                                                 class_num, batch_size, sess, proby_axial, x_axial)

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
            if(config_test.get('save_binary_seg', False)):
                out_label = post_process_on_edt(pred).astype(np.int16)
            elif(config_test.get('save_predicted_map', False)):
                save_array_as_nifty_volume((pred).astype(np.int16), os.path.join("./ResultTem", "PredictedMap", emrbyo_name, img_names))
            else:
                sess.close()
                raise Exception('No distance map or binary segmentation is to be saved')


            # ==============================================================
            #               save prediction results as *.nii.gz
            # ==============================================================
            test_time.append(time.time() - t0)
            final_label = np.zeros(temp_size, out_label.dtype)
            final_label = set_crop_to_volume(final_label, temp_bbox[0], temp_bbox[1], out_label)
            final_label = transpose_volumes_reverse(final_label, slice_direction)
            save_file = os.path.join(save_folder,emrbyo_name, emrbyo_name + "_" + img_names.split(".")[0].split("_")[1] + "_segMemb.nii.gz")
            save_array_as_nifty_volume(final_label, save_file) # os.path.join(save_folder, one_embryo, one_embryo + "_" + tp_str.zfill(3) + "_cell.nii.gz")

            print(save_file)
        test_time = np.asarray(test_time)
        print('test time', test_time.mean())
        np.savetxt(save_folder + '/test_time.txt', test_time)
        sess.close()


    # ==============================================================
    #               Post processing (binary membrane --> isolated SegCell)
    # ==============================================================
    if config_test.get('post_process', False):
        config_post = {}
        config_post['segdata'] = config['segdata']
        config_post['segdata']['embryos'] = config_data['data_names']
        config_post['debug'] = config['debug']
        config_post['result'] = config['result']

        config_post['segdata']['membseg_path'] = config_data['save_folder']
        config_post['result']['postseg_folder'] = config_data['save_folder'] + 'Postseg'
        config_post['result']['nucleus_filter'] = config_test['nucleus_filter']
        if not os.path.isdir(config_post['result']['postseg_folder'] ):
            os.makedirs(config_post['result']['postseg_folder'] )

        post_process(config_post)


if __name__ == '__main__':
    st = time.time()
    if (len(sys.argv) != 2):
        raise Exception("Invalid number of inputs")
    config_file = str(sys.argv[1])
    assert (os.path.isfile(config_file))
    test(config_file)

