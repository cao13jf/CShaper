from PyQt5.QtCore import pyqtSignal, QThread

import os
import gc
import time
import glob
import torch
import shutil
import pickle
import warnings
import traceback
import numpy as np
import pandas as pd
from tqdm import tqdm
from numba import cuda
from Util.DMFNet16 import EDTDMFNet
from PIL import Image
import tensorflow as tf
from skimage.transform import resize
from multiprocessing import Pool, cpu_count, Lock
from tensorflow.keras import regularizers
from torch.utils.data import DataLoader
from torch.optim import Adam


# User defined library
from Util.torch_dataset import Memb3DDataset
from Util.sampler import CycleSampler
from Util.ProcessLib import segment_membrane
from Util.data_utils import init_fn, adjust_learning_rate, mse_loss
from shape_analysis import init, cell_graph_network, construct_stat_embryo, analyse_seg, assemble_result, construct_celltree
from Util.preprocess_lib import stack_memb_slices, save_nuc_seg, stack_nuc_slices
from Util.segmentation_post_process import run_post, save_nii
from Util.loss_function import weighted_one_hot_loss
from Util.data_process import delete_isolate_labels, set_crop_to_volume, post_process_on_edt, transpose_volumes_reverse, save_array_as_nifty_volume
from Util.data_loader import DataGene, transpose_volumes
from Util.train_test_func import predict_full_volume, prediction_fusion
from Util.DMapNetUpdated import DMapNetCompiled, ProgressBar

warnings.filterwarnings("ignore")

USE_GPU = len(tf.config.list_physical_devices('GPU')) > 0


class PreprocessThread(QThread):
    signal = pyqtSignal(bool, str, str)
    process = pyqtSignal(str, int, int)

    def __init__(self, config={}):
        self.config = config
        self.flag = None
        self.mpPool = None
        super(PreprocessThread, self).__init__()

    def __del__(self):
        self.wait()

    def threadflag(self, flag):
        self.flag = flag

    def combine_slices(self, process, config):
        """
        Combine slices into stack images
        :param config: parameters
        :return:
        """
        # signal.emit(True,'sss')
        num_slice = config["num_slice"]
        embryo_names = config["embryo_names"]
        max_time = config["max_time"]
        xy_res = config["xy_resolution"]
        z_res = config["z_resolution"]
        reduce_ratio = config["reduce_ratio"]
        raw_folder = config["raw_folder"]
        stack_folder = os.path.join(config["project_folder"], "RawStack")
        lineage_file = config.get("lineage_file", None)
        number_dictionary = config["number_dictionary"]

        # get output size
        raw_memb_files = glob.glob(os.path.join(raw_folder, embryo_names[0], "tifR", "*.tif"))
        raw_size = list(np.asarray(Image.open(raw_memb_files[0])).shape) + [int(num_slice * z_res / xy_res)]
        out_size = [int(i * reduce_ratio) for i in raw_size]
        out_res = [res * x / y for res, x, y in zip([xy_res, xy_res, xy_res], raw_size, out_size)]

        # multiprocessing
        self.mpPool = Pool(cpu_count() - 1)
        for embryo_name in embryo_names:
            # save nucleus
            origin_files = glob.glob(os.path.join(raw_folder, embryo_name, "tif", "*.tif"))
            origin_files = sorted(origin_files)
            target_folder = os.path.join(stack_folder, embryo_name, "RawNuc")
            if not os.path.isdir(target_folder):
                os.makedirs(target_folder)

            configs = []
            for tp in range(1, max_time + 1):
                configs.append((origin_files, target_folder, embryo_name, tp, out_size, num_slice, out_res))

            self.flag = True
            for idx, _ in enumerate(tqdm(self.mpPool.imap_unordered(stack_nuc_slices, configs), total=len(configs),
                                         desc="1/3 Stack nucleus of {}".format(embryo_name))):
                process.emit('1/3 Stack nucleus', idx, max_time)
                if not self.flag:
                    self.mpPool.close()
                    return 0

            # save membrane
            origin_files = glob.glob(os.path.join(raw_folder, embryo_name, "tifR", "*.tif"))
            origin_files = sorted(origin_files)
            target_folder = os.path.join(stack_folder, embryo_name, "RawMemb")
            if not os.path.isdir(target_folder):
                os.makedirs(target_folder)

            configs = []
            for tp in range(1, max_time + 1):
                configs.append((origin_files, target_folder, embryo_name, tp, out_size, num_slice, out_res))
            self.flag = True
            for idx, _ in enumerate(tqdm(self.mpPool.imap_unordered(stack_memb_slices, configs), total=len(configs),
                                         desc="2/3 Stack membrane of {}".format(embryo_name))):
                process.emit('2/3 Stacking membrane', idx, max_time)
                if not self.flag:
                    self.mpPool.close()
                    return 0

            # save nucleus
            if lineage_file is not None:
                target_folder = os.path.join(stack_folder, embryo_name, "SegNuc")
                if not os.path.isdir(target_folder):
                    os.makedirs(target_folder)
                pd_lineage = pd.read_csv(lineage_file, dtype={"cell": str,
                                                              "time": np.int16,
                                                              "z": np.float32,
                                                              "x": np.float32,
                                                              "y": np.float32})

                pd_number = pd.read_csv(number_dictionary, names=["name", "label"])
                number_dict = pd.Series(pd_number.label.values, index=pd_number.name).to_dict()

                configs = []
                for tp in range(1, max_time + 1):
                    configs.append((embryo_name, number_dict, pd_lineage, tp, raw_size, out_size, out_res,
                                    xy_res / z_res, target_folder))
                self.flag = True
                for idx, _ in enumerate(tqdm(self.mpPool.imap_unordered(save_nuc_seg, configs), total=len(configs),
                                             desc="3/3 Construct nucleus location of {}".format(embryo_name))):
                    process.emit('3/3 Constructing nucleus location', idx, max_time)
                    if not self.flag:
                        self.mpPool.close()
                        return 0
                shutil.copy(lineage_file, os.path.join(stack_folder, embryo_name))
        return 1

    def run(self):
        try:
            sin = self.combine_slices(self.process, self.config)
            if sin == 1:
                self.mpPool.close()
                self.signal.emit(True, 'Preprocess', 'Preprocess Completed!')
        except Exception:
            if self.mpPool:
                self.mpPool.close()
            self.signal.emit(False, 'Preprocess', traceback.format_exc())


class SegmentationThread(QThread):
    signal = pyqtSignal(bool, str, str)
    process = pyqtSignal(str, int, int)

    def __init__(self, config={}):
        self.config = config
        self.flag = None
        self.mpPool = None
        super(SegmentationThread, self).__init__()

    def __del__(self):
        self.wait()

    def threadflag(self, flag):
        self.flag = flag

    def test_cmap(self, process, config):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        config_data = config['data']
        config_net = config.get('network', None)
        config_test = config['testing']
        batch_size = config_test.get('batch_size', 5)

        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        # get membrane binary shell
        test_folder = dict(root=config_data['save_folder'], has_label=False)
        # =============================================================
        #  construct network model
        # =============================================================
        model = EDTDMFNet(in_channels=1, n_first=32, conv_channels=64, groups=16, norm="in", out_class=1)
        check_point = torch.load(config_net["model_file"], map_location=device)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in check_point["state_dict"].items():
            if "module." in k:
                name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        model.to(device)

        data_root = config_data['data_root'] if type(config_data['data_root']) is list else [config_data['data_root']]  # Save as list
        data_names = config_data.get('data_names', None)
        augmentations = "Compose([Resize((256,352,224)),NumpyType((np.float32, np.float32)),])"
        # =============================================================
        #    set data loader
        # =============================================================
        test_set = Memb3DDataset(root=data_root, membrane_names=data_names, for_train=False, transforms=augmentations,
                                 return_target=False, suffix="*.nii.gz", max_times=config_data["max_time"])
        test_loader = DataLoader(
            dataset=test_set,
            batch_size=1,
            shuffle=False,
            num_workers=10,
            pin_memory=True
        )

        # =============================================================
        #  begin prediction
        # =============================================================
        #  Prepare (or clear) in order to update all files
        save_folder = config_data['save_folder']
        for embryo_name in data_names:
            if os.path.isdir(os.path.join(save_folder, embryo_name)):
                shutil.rmtree(os.path.join(save_folder, embryo_name))

        # the file will save in the segMemb folder
        data_number = config_data.get('max_time', 100) * len(config_data["data_names"])
        with torch.no_grad():
            model.eval()
            runtimes = []
            for i, data in enumerate(tqdm(test_loader, desc="Getting binary membrane:")):
                process.emit("1/2 Extracting binary membrane", i, data_number)
                x, nuc = data[0:2]  #
                #  go through the network
                x = x.to(device)
                start_time = time.time()
                pred_bin = model(x)
                elapsed_time = time.time() - start_time
                runtimes.append(elapsed_time)
                #  Regression only has one channel
                if pred_bin.shape[1] > 1:
                    pred_bin = delete_isolate_labels(pred_bin)
                    pred_bin = pred_bin.argmax(1)  # [channel, height, width, depth]

                #  binary prediction
                if torch.cuda.is_available():
                    pred_bin = pred_bin.cpu().numpy()
                else:
                    pred_bin = pred_bin.numpy()
                pred_bin = pred_bin.squeeze()  # .transpose([1, 2, 0])  # from z, x, y to x, y ,z
                pred_bin = resize(pred_bin.astype(float), test_set.size, mode='constant', cval=0, order=0, anti_aliasing=False)
                save_file = os.path.join(save_folder, embryo_name, embryo_name + "_" + test_set.names[i].split(".")[0].split("_")[1] + "_segMemb.nii.gz")
                save_array_as_nifty_volume((pred_bin * 256).astype(np.int16), save_file)  # pred_bin is range(0,1)

        # ============================
        # Membrane distance map to cells
        # ============================
        embryo_mask = None
        file_names = glob.glob(os.path.join(save_folder, embryo_name, '*.nii.gz'))
        parameters = []

        for file_name in file_names:
            parameters.append([embryo_name, file_name, embryo_mask, save_folder])
            # segment_membrane([embryo_name, file_name, embryo_mask])
        self.mpPool = Pool(cpu_count() - 1)
        self.flag = True
        for idx, _ in enumerate(tqdm(self.mpPool.imap_unordered(segment_membrane, parameters), total=len(parameters),
                      desc="{} edt cell membrane --> single cell instance".format(embryo_name))):
            process.emit('2/2 Segmenting distance map', idx, config_data['max_time'])
            if not self.flag:
                self.mpPool.close()
                return 0

        if not self.mpPool:
            self.mpPool.close()
        # if config_test.get('post_process', False):
        #     config_post = {}
        #     config_post['segdata'] = config['segdata']
        #     config_post['segdata']['embryos'] = config_data['data_names']
        #     config_post['debug'] = config['debug']
        #     config_post['result'] = {}
        #
        #     config_post['segdata']['membseg_path'] = config_data['save_folder']
        #     config_post['result']['postseg_folder'] = config_data['save_folder'] + 'Postseg'
        #     config_post['result']['nucleus_filter'] = config_test['nucleus_filter']
        #     config_post['result']['nucleus_as_seed'] = config_test['nucleus_as_seed']
        #     config_post['segdata']['max_time'] = config["para"]["max_time"]
        #     if not os.path.isdir(config_post['result']['postseg_folder']):
        #         os.makedirs(config_post['result']['postseg_folder'])
        #
        #     self.post_process(process, config_post)
        return 1
    def test(self, process, config):

        # group parameters
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
            #               2, Construct computation graph
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
            net_axial.load_weights(config_net["model_file"].split("ckpt")[0] + "ckpt")

            temp_imgs = transpose_volumes(temp_imgs, slice_direction='sagittal')
            [D, H, W] = temp_imgs.shape
            Hx = max(int((H + 3) / 4) * 4, data_shape[1])
            Wx = max(int((W + 3) / 4) * 4, data_shape[2])
            data_slice = data_shape[0]
            full_data_shape = [data_slice, Hx, Wx, data_shape[-1]]
            net_sagittal = DMapNetCompiled(input_size=full_data_shape,
                                           num_classes=config_data['edt_discrete_num'],
                                           activation="relu")
            net_sagittal.load_weights(config_net["model_file"].split("ckpt")[0] + "ckpt")

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

            self.flag = True
            for i in tqdm(range(0, data_number), desc='Extracting binary membrane'):
                process.emit("1/2 Extracting binary membrane", i, data_number)
                [temp_imgs, img_names, emrbyo_name, temp_bbox, temp_size] = dataloader.get_image_data_with_name(i)
                t0 = time.time()

                temp_img_sagittal = transpose_volumes(temp_imgs, slice_direction="sagittal")
                prob_sagittal = predict_full_volume(temp_img_sagittal, data_shape[:-1], label_shape[:-1],
                                                    data_shape[-1],
                                                    class_num, batch_size, net_sagittal)
                temp_img_axial = transpose_volumes(temp_imgs, slice_direction='axial')
                prob_axial = predict_full_volume(temp_img_axial, data_shape[:-1], label_shape[:-1], data_shape[-1],
                                                 class_num, batch_size, net_axial)

                # If the prediction is one-hot tensor, use np.argmax to get the indices of the maximumm. That indice is used as label
                if (label_edt_discrete):  # If we hope to predict discrete EDT map, argmax on the indices depth
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
                # temp_bbox[0] = [temp_bbox[0][1], temp_bbox[0][2], temp_bbox[0][0]]
                # temp_bbox[1] = [temp_bbox[1][1], temp_bbox[1][2], temp_bbox[1][0]]
                final_label = set_crop_to_volume(final_label, temp_bbox[0], temp_bbox[1], out_label)
                # final_label = transpose_volumes_reverse(final_label, slice_direction)
                save_file = os.path.join(save_folder, emrbyo_name,
                                         emrbyo_name + "_" + img_names.split(".")[0].split("_")[1] + "_segMemb.nii.gz")
                save_array_as_nifty_volume(final_label,
                                           save_file)  # os.path.join(save_folder, one_embryo, one_embryo + "_" + tp_str.zfill(3) + "_cell.nii.gz")
                if not self.flag:
                    del net_axial, net_sagittal, dataloader
                    return 0

            del net_axial, net_sagittal, dataloader

        # ==============================================================
        #               Post processing (binary membrane --> isolated SegCell)
        # ==============================================================
        if config_test.get('post_process', False):
            config_post = {}
            config_post['segdata'] = config['segdata']
            config_post['segdata']['embryos'] = config_data['data_names']
            config_post['debug'] = config['debug']
            config_post['result'] = {}

            config_post['segdata']['membseg_path'] = config_data['save_folder']
            config_post['result']['postseg_folder'] = config_data['save_folder'] + 'Postseg'
            config_post['result']['nucleus_filter'] = config_test['nucleus_filter']
            config_post['result']['nucleus_as_seed'] = config_test['nucleus_as_seed']
            config_post['segdata']['max_time'] = config["para"]["max_time"]
            if not os.path.isdir(config_post['result']['postseg_folder']):
                os.makedirs(config_post['result']['postseg_folder'])

            self.post_process(process, config_post)
            return 1

    def post_process(self, process, config):

        config_segdata = config['segdata']
        membseg_path = config_segdata['membseg_path']

        # get all files under the path
        if config_segdata.get('embryos', None) is not None:
            embryo_names = config_segdata['embryos']
        else:
            embryo_names = os.listdir(membseg_path)
            embryo_names = [embryo_name for embryo_name in embryo_names if 'plc' in embryo_name.lower()]
        membseg_path = config_segdata['membseg_path']
        parameters = []
        for one_embryo in embryo_names:
            if os.path.isdir(os.path.join(config['result']['postseg_folder'], one_embryo)):
                shutil.rmtree(os.path.join(config['result']['postseg_folder'], one_embryo))
                os.makedirs(os.path.join(config['result']['postseg_folder'], one_embryo))
            else:
                os.path.join(config['result']['postseg_folder'], one_embryo)

            file_names = glob.glob(os.path.join(membseg_path, one_embryo, '*.nii.gz'))
            file_names.sort()
            for file_name in file_names[:config_segdata['max_time'] + 1]:
                parameters.append([one_embryo, file_name, config])

        self.mpPool = Pool(2)
        self.flag = True
        for idx, _ in enumerate(tqdm(self.mpPool.imap_unordered(run_post, parameters), total=len(parameters),
                                     desc="Segmenting binary membrane to cells")):
            #  2 / 2 Process name: `Segment binary membrane to cells`;  current status: `idx`; final status: `len(parameters)`;
            process.emit('2/2 Segmenting binary membrane', idx, config_segdata['max_time'])
            if not self.flag:
                self.mpPool.close()
                return 0

    def run(self):
        try:
            # TODO
            if self.config["network"]["framework_name"] == "CShaper":
                sin = self.test(self.process, self.config)
            elif self.config["network"]["framework_name"] == "CMap":
                sin = self.test_cmap(self.process, self.config)

            if sin == 1:
                self.mpPool.close()
                self.signal.emit(True, 'Segmentation', 'Segmentation Completed!')
                # if self.config["network"]["framework_name"] == "CShaper":
                #     if USE_GPU:
                #         device = cuda.get_current_device()
                #         device.reset()
                #         cuda.close()
        except Exception:
            if self.mpPool:
                self.mpPool.close()
            # if self.config["network"]["framework_name"] == "CShaper":
            #     if USE_GPU:
            #         device = cuda.get_current_device()
            #         device.reset()
            #         cuda.close()
            self.signal.emit(False, 'Segmentation', traceback.format_exc())


class AnalysisThread(QThread):
    signal = pyqtSignal(bool, str, str)
    process = pyqtSignal(str, int, int)

    def __init__(self, config={}):
        self.config = config
        self.flag = None
        self.mpPool = None
        super(AnalysisThread, self).__init__()

    def __del__(self):
        self.wait()

    def threadflag(self, flag):
        self.flag = flag

    def run_shape_analysis(self, process, config):
        '''
        Extract the cell tree structure from the aceTree file
        :param acetree_file:  file name of the embryo acetree file
        :param max_time:  the maximum time point in the tree.
        :return :
        '''
        global max_time
        global cell_tree
        ## construct lineage tree whose nodes contain the time points that cell exist (based on nucleus).
        acetree_file = config['acetree_file']
        cell_tree, max_time = construct_celltree(acetree_file, config)
        save_file_name = os.path.join(config['stat_folder'], config['embryo_name'] + '_time_tree.txt')
        with open(save_file_name, 'wb') as f:
            pickle.dump(cell_tree, f)
        ## Parallel computing for the cell relation graph
        if not os.path.isdir(os.path.join(config["project_folder"], 'TemCellGraph')):
            os.makedirs(os.path.join(config["project_folder"], 'TemCellGraph'))

        pd_number = pd.read_csv(config["number_dictionary"], names=["name", "label"])
        number_dict = pd.Series(pd_number.label.values, index=pd_number.name).to_dict()

        # ========================================================
        #       sementing TPs in a parallel way
        # ========================================================
        file_lock = Lock()  # |-----> for change treelib files
        self.mpPool = Pool(cpu_count() - 1, initializer=init, initargs=(file_lock,))
        configs = []
        config["cell_tree"] = cell_tree
        for itime in tqdm(range(1, max_time + 1), desc="Composing configs"):
            config['time_point'] = itime
            configs.append(config.copy())
        embryo_name = config["embryo_names"][0]
        self.flag = True
        for idx, _ in enumerate(tqdm(self.mpPool.imap_unordered(cell_graph_network, configs), total=len(configs),
                                     desc="Naming {} segmentations".format(embryo_name))):
            process.emit('1/3 Naming segmentation', idx, max_time)
            if not self.flag:
                self.mpPool.close()
                return 0

        # ========================================================
        #       Combine previous TPs
        # ========================================================
        # ## In order to make use of parallel computing, the global vairable stat_embryo cannot be shared between different processor,
        # #  so we need to store all one-embryo reults as temporary files, which will be assembled finally. After that, these temporary
        # #  Data can be deleted.
        construct_stat_embryo(cell_tree,
                              max_time)  # initilize the shape matrix which is use to store the shape series information
        self.flag = True
        for itime in tqdm(range(1, max_time + 1), desc='assembling {} result'.format(embryo_name)):
            process.emit('2/3 Assembling results', itime, max_time)
            file_name = os.path.join(config["project_folder"], 'TemCellGraph', config['embryo_name'],
                                     config['embryo_name'] + '_T' + str(itime) + '.txt')
            with open(file_name, 'rb') as f:
                cell_graph = pickle.load(f)
                stat_embryo = assemble_result(cell_graph, itime, number_dict)
            if not self.flag:
                self.mpPool.close()
                return 0

        # =======================================================
        # Combine all surfaces and volumes in one single file
        # =======================================================
        # combien all volume and surface informace
        if not os.path.isdir(os.path.join(config['stat_folder'], embryo_name)):
            os.makedirs(os.path.join(config['stat_folder'], embryo_name))
        volume_lists = []
        surface_lists = []
        self.flag = True
        for t in tqdm(range(1, max_time + 1), desc="Generate surface and volume {}".format(embryo_name.split('/')[-1])):
            process.emit('3/3 Generating surface and volume', t, max_time)
            nucleus_loc_file = os.path.join(config["save_nucleus_folder"], embryo_name,
                                            os.path.basename(embryo_name) + "_" + str(t).zfill(3) + "_nucLoc" + ".csv")
            pd_loc = pd.read_csv(nucleus_loc_file)
            cell_volume_surface = pd_loc[["nucleus_name", "volume", "surface"]]
            cell_volume_surface = cell_volume_surface.set_index("nucleus_name")
            volume_lists.append(cell_volume_surface["volume"].to_frame().T.dropna(axis=1))
            surface_lists.append(cell_volume_surface["surface"].to_frame().T.dropna(axis=1))
            if not self.flag:
                self.mpPool.close()
                return 0

        volume_stat = pd.concat(volume_lists, keys=range(1, max_time + 1), ignore_index=True, axis=0, sort=False,
                                join="outer")
        surface_stat = pd.concat(surface_lists, keys=range(1, max_time + 1), ignore_index=True, axis=0, sort=False,
                                 join="outer")
        cell_numbers = np.count_nonzero(~np.isnan(volume_stat.to_numpy()), axis=1).astype(np.uint16).tolist()

        volume_stat = volume_stat.set_index(pd.Index(cell_numbers).astype("int64"))
        volume_stat.index.name = "Cell Number"
        surface_stat = surface_stat.set_index(pd.Index(cell_numbers).astype("int64"))
        surface_stat.index.name = "Cell Number"
        volume_stat.to_csv(
            os.path.join(config["stat_folder"], embryo_name, embryo_name.split('/')[-1] + "_volume" + '.csv'))
        surface_stat.to_csv(
            os.path.join(config["stat_folder"], embryo_name, embryo_name.split('/')[-1] + "_surface" + '.csv'))

        if config['delete_tem_file']:  # If need to delete temporary files.
            shutil.rmtree(os.path.join(config["project_folder"], 'TemCellGraph'))
        stat_embryo = stat_embryo.loc[:, ((stat_embryo != 0) & (~np.isnan(stat_embryo))).any(axis=0)]
        save_file_name_csv = os.path.join(config['stat_folder'], embryo_name, config['embryo_name'] + '_contact.csv')
        stat_embryo = stat_embryo.set_index(pd.Index(cell_numbers))
        stat_embryo.index.name = "Cell Number"
        stat_embryo.to_csv(save_file_name_csv)
        return 1

    def run_shape_analysis_nolineage(self, process, config):
        target_folder = os.path.join(config["project_folder"], "RawStack", config["embryo_names"][0], "RawMemb")
        max_time = len(os.listdir(target_folder))

        self.mpPool = Pool(cpu_count() - 1)
        configs = []
        for itime in tqdm(range(1, max_time + 1), desc="Compose configs"):
            config['time_point'] = itime
            configs.append(config.copy())

        embryo_name = config["embryo_names"][0]
        self.flag = True
        for idx, _ in enumerate(tqdm(self.mpPool.imap_unordered(analyse_seg, configs), total=len(configs),
                                     desc="Naming {} segmentations".format(embryo_name))):
            process.emit('Collecting surface and volume', idx, max_time)
            if not self.flag:
                return 0
        return 1

    def run(self):
        try:
            para_config = self.config['para']
            para_config["data_folder"] = os.path.join(para_config["project_folder"], "RawStack")
            para_config["save_nucleus_folder"] = os.path.join(para_config["project_folder"], "NucleusLoc")
            para_config["seg_folder"] = os.path.join(para_config["project_folder"], "CellMembranePostseg")
            para_config["stat_folder"] = os.path.join(para_config["project_folder"], "StatShape")
            para_config["delete_tem_file"] = False

            if not os.path.isdir(para_config['stat_folder']):
                os.makedirs(para_config['stat_folder'])
            # Get the size of the figure
            example_embryo_folder = os.path.join(para_config["raw_folder"], para_config["embryo_names"][0], "tif")
            example_img_file = glob.glob(os.path.join(example_embryo_folder, "*.tif"))
            raw_size = [para_config["num_slice"]] + list(np.asarray(Image.open(example_img_file[0])).shape)
            para_config["image_size"] = [raw_size[0], raw_size[2], raw_size[1]]

            para_config["embryo_name"] = para_config["embryo_names"][0]
            para_config["acetree_file"] = para_config["lineage_file"]
            if not os.path.isdir(os.path.join(para_config['save_nucleus_folder'], para_config['embryo_name'])):
                os.makedirs(os.path.join(para_config['save_nucleus_folder'], para_config['embryo_name']))
            else:
                shutil.rmtree(os.path.join(para_config['save_nucleus_folder'], para_config['embryo_name']))
                os.makedirs(os.path.join(para_config['save_nucleus_folder'], para_config['embryo_name']))
            if para_config['lineage_file'] == '':
                sin = self.run_shape_analysis_nolineage(self.process, para_config)
            else:
                sin = self.run_shape_analysis(self.process, para_config)
            if sin == 1:
                self.mpPool.close()
                self.signal.emit(True, 'Analysis', 'Analysis Completed!')
        except Exception:
            if self.mpPool:
                self.mpPool.close()
            self.signal.emit(False, 'Analysis', traceback.format_exc())


class RunAllThread(QThread):
    signal = pyqtSignal(bool, str, str)
    process = pyqtSignal(str, int, int)
    segmentation = pyqtSignal(str, int, int)
    analysis = pyqtSignal(str, int, int)

    def __init__(self, config={}):
        self.config = config
        self.flag = None
        self.mpPool = None
        super(RunAllThread, self).__init__()

    def __del__(self):
        self.wait()

    def threadflag(self, flag):
        self.flag = flag

    def combine_slices(self, process, config):
        """
        Combine slices into stack images
        :param config: parameters
        :return:
        """
        # signal.emit(True,'sss')
        num_slice = config["num_slice"]
        embryo_names = config["embryo_names"]
        max_time = config["max_time"]
        xy_res = config["xy_resolution"]
        z_res = config["z_resolution"]
        reduce_ratio = config["reduce_ratio"]
        raw_folder = config["raw_folder"]
        stack_folder = os.path.join(config["project_folder"], "RawStack")
        lineage_file = config.get("lineage_file", None)
        number_dictionary = config["number_dictionary"]

        # get output size
        raw_memb_files = glob.glob(os.path.join(raw_folder, embryo_names[0], "tifR", "*.tif"))
        raw_size = list(np.asarray(Image.open(raw_memb_files[0])).shape) + [int(num_slice * z_res / xy_res)]
        out_size = [int(i * reduce_ratio) for i in raw_size]
        out_res = [res * x / y for res, x, y in zip([xy_res, xy_res, xy_res], raw_size, out_size)]

        # multiprocessing
        self.mpPool = Pool(cpu_count() - 1)
        for embryo_name in embryo_names:
            # save nucleus
            origin_files = glob.glob(os.path.join(raw_folder, embryo_name, "tif", "*.tif"))
            origin_files = sorted(origin_files)
            target_folder = os.path.join(stack_folder, embryo_name, "RawNuc")
            if not os.path.isdir(target_folder):
                os.makedirs(target_folder)

            configs = []
            for tp in range(1, max_time + 1):
                configs.append((origin_files, target_folder, embryo_name, tp, out_size, num_slice, out_res))

            self.flag = True
            for idx, _ in enumerate(tqdm(self.mpPool.imap_unordered(stack_nuc_slices, configs), total=len(configs),
                                         desc="1/3 Stack nucleus of {}".format(embryo_name))):
                self.process.emit('1/3 Stack nucleus', idx, max_time)
                if not self.flag:
                    self.mpPool.close()
                    return 0

            # save membrane
            origin_files = glob.glob(os.path.join(raw_folder, embryo_name, "tifR", "*.tif"))
            origin_files = sorted(origin_files)
            target_folder = os.path.join(stack_folder, embryo_name, "RawMemb")
            if not os.path.isdir(target_folder):
                os.makedirs(target_folder)

            configs = []
            for tp in range(1, max_time + 1):
                configs.append((origin_files, target_folder, embryo_name, tp, out_size, num_slice, out_res))
            self.flag = True
            for idx, _ in enumerate(tqdm(self.mpPool.imap_unordered(stack_memb_slices, configs), total=len(configs),
                                         desc="2/3 Stack membrane of {}".format(embryo_name))):
                self.process.emit('2/3 Stacking membrane', idx, max_time)
                if not self.flag:
                    self.mpPool.close()
                    return 0

            # save nucleus
            if lineage_file is not None:
                target_folder = os.path.join(stack_folder, embryo_name, "SegNuc")
                if not os.path.isdir(target_folder):
                    os.makedirs(target_folder)
                pd_lineage = pd.read_csv(lineage_file, dtype={"cell": str,
                                                              "time": np.int16,
                                                              "z": np.float32,
                                                              "x": np.float32,
                                                              "y": np.float32})

                pd_number = pd.read_csv(number_dictionary, names=["name", "label"])
                number_dict = pd.Series(pd_number.label.values, index=pd_number.name).to_dict()

                configs = []
                for tp in range(1, max_time + 1):
                    configs.append((embryo_name, number_dict, pd_lineage, tp, raw_size, out_size, out_res,
                                    xy_res / z_res, target_folder))
                self.flag = True
                for idx, _ in enumerate(tqdm(self.mpPool.imap_unordered(save_nuc_seg, configs), total=len(configs),
                                             desc="3/3 Construct nucleus location of {}".format(embryo_name))):
                    self.process.emit('3/3 Constructing nucleus location', idx, max_time)
                    if not self.flag:
                        self.mpPool.close()
                        return 0
                shutil.copy(lineage_file, os.path.join(stack_folder, embryo_name))
        if self.mpPool:
            self.mpPool.close()
        return 1

    def test_cmap(self, process, config):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        config_data = config['data']
        config_net = config.get('network', None)
        config_test = config['testing']
        batch_size = config_test.get('batch_size', 5)

        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        # get membrane binary shell
        test_folder = dict(root=config_data['save_folder'], has_label=False)
        # =============================================================
        #  construct network model
        # =============================================================
        model = EDTDMFNet(in_channels=1, n_first=32, conv_channels=64, groups=16, norm="in", out_class=1)
        model.to(device)
        check_point = torch.load(config_net["model_file"])
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in check_point["state_dict"].items():
            if "module." in k:
                name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

        data_root = config_data['data_root'] if type(config_data['data_root']) is list else [
            config_data['data_root']]  # Save as list
        data_names = config_data.get('data_names', None)
        augmentations = "Compose([Resize((256,352,224)),NumpyType((np.float32, np.float32)),])"
        # =============================================================
        #    set data loader
        # =============================================================
        test_set = Memb3DDataset(root=data_root, membrane_names=data_names, for_train=False, transforms=augmentations,
                                 return_target=False, suffix="*.nii.gz", max_times=config_data["max_time"])
        test_loader = DataLoader(
            dataset=test_set,
            batch_size=1,
            shuffle=False,
            num_workers=10,
            pin_memory=True
        )

        # =============================================================
        #  begin prediction
        # =============================================================
        #  Prepare (or clear) in order to update all files
        save_folder = config_data['save_folder']
        for embryo_name in data_names:
            if os.path.isdir(os.path.join(save_folder, embryo_name)):
                shutil.rmtree(os.path.join(save_folder, embryo_name))

        # the file will save in the segMemb folder
        data_number = config_data.get('max_time', 100) * len(config_data["data_names"])
        with torch.no_grad():
            model.eval()
            runtimes = []
            for i, data in enumerate(tqdm(test_loader, desc="Getting binary membrane:")):
                self.segmentation.emit("1/2 Extracting binary membrane", i, data_number)
                x, nuc = data[0:2]  #
                x = x.to(device)
                #  go through the network
                start_time = time.time()
                pred_bin = model(x)
                elapsed_time = time.time() - start_time
                runtimes.append(elapsed_time)
                #  Regression only has one channel
                if pred_bin.shape[1] > 1:
                    pred_bin = delete_isolate_labels(pred_bin)
                    pred_bin = pred_bin.argmax(1)  # [channel, height, width, depth]

                #  binary prediction
                pred_bin = pred_bin.cpu().numpy()
                pred_bin = pred_bin.squeeze()  #.transpose([1, 2, 0])  # from z, x, y to x, y ,z
                pred_bin = resize(pred_bin.astype(float), test_set.size, mode='constant', cval=0, order=0,
                                  anti_aliasing=False)
                save_file = os.path.join(save_folder, embryo_name,
                                         embryo_name + "_" + test_set.names[i].split(".")[0].split("_")[
                                             1] + "_segMemb.nii.gz")
                save_array_as_nifty_volume((pred_bin * 256).astype(np.int16), save_file)  # pred_bin is range(0,1)

        # ============================
        # Membrane distance map to cells
        # ============================
        embryo_mask = None
        file_names = glob.glob(os.path.join(save_folder, embryo_name, '*.nii.gz'))
        parameters = []

        for file_name in file_names:
            parameters.append([embryo_name, file_name, embryo_mask, save_folder])
            # segment_membrane([embryo_name, file_name, embryo_mask])
        self.mpPool = Pool(cpu_count() - 1)
        self.flag = True
        for idx, _ in enumerate(tqdm(self.mpPool.imap_unordered(segment_membrane, parameters), total=len(parameters),
                                     desc="{} edt cell membrane --> single cell instance".format(embryo_name))):
            self.segmentation.emit('2/2 Segmenting distance map', idx, config_data['max_time'])
            if not self.flag:
                self.mpPool.close()
                return 0

        if not self.mpPool:
            self.mpPool.close()
        return 1

    def test(self, process, config):

        # group parameters
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
            #               2, Construct computation graph
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
            net_axial.load_weights(config_net["model_file"])

            temp_imgs = transpose_volumes(temp_imgs, slice_direction='sagittal')
            [D, H, W] = temp_imgs.shape
            Hx = max(int((H + 3) / 4) * 4, data_shape[1])
            Wx = max(int((W + 3) / 4) * 4, data_shape[2])
            data_slice = data_shape[0]
            full_data_shape = [data_slice, Hx, Wx, data_shape[-1]]
            net_sagittal = DMapNetCompiled(input_size=full_data_shape,
                                           num_classes=config_data['edt_discrete_num'],
                                           activation="relu")
            net_sagittal.load_weights(config_net["model_file"])

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

            self.flag = True
            for i in tqdm(range(0, data_number), desc='Extracting binary membrane'):
                self.segmentation.emit("1/2 Extracting binary membrane", i, data_number)
                [temp_imgs, img_names, emrbyo_name, temp_bbox, temp_size] = dataloader.get_image_data_with_name(i)
                t0 = time.time()

                temp_img_sagittal = transpose_volumes(temp_imgs, slice_direction="sagittal")
                prob_sagittal = predict_full_volume(temp_img_sagittal, data_shape[:-1], label_shape[:-1],
                                                    data_shape[-1],
                                                    class_num, batch_size, net_sagittal)
                temp_img_axial = transpose_volumes(temp_imgs, slice_direction='axial')
                prob_axial = predict_full_volume(temp_img_axial, data_shape[:-1], label_shape[:-1], data_shape[-1],
                                                 class_num, batch_size, net_axial)

                # If the prediction is one-hot tensor, use np.argmax to get the indices of the maximumm. That indice is used as label
                if (label_edt_discrete):  # If we hope to predict discrete EDT map, argmax on the indices depth
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
                # temp_bbox[0] = [temp_bbox[0][1], temp_bbox[0][2], temp_bbox[0][0]]
                # temp_bbox[1] = [temp_bbox[1][1], temp_bbox[1][2], temp_bbox[1][0]]
                final_label = set_crop_to_volume(final_label, temp_bbox[0], temp_bbox[1], out_label)
                # final_label = transpose_volumes_reverse(final_label, slice_direction)
                save_file = os.path.join(save_folder, emrbyo_name,
                                         emrbyo_name + "_" + img_names.split(".")[0].split("_")[1] + "_segMemb.nii.gz")
                save_array_as_nifty_volume(final_label,
                                           save_file)  # os.path.join(save_folder, one_embryo, one_embryo + "_" + tp_str.zfill(3) + "_cell.nii.gz")
                if not self.flag:
                    del net_axial, net_sagittal, dataloader
                    return 0

            del net_axial, net_sagittal, dataloader

        if not self.mpPool:
            self.mpPool.close()
        if USE_GPU:
            device = cuda.get_current_device()
            device.reset()
            cuda.close()

        # ==============================================================
        #               Post processing (binary membrane --> isolated SegCell)
        # ==============================================================
        if config_test.get('post_process', False):
            config_post = {}
            config_post['segdata'] = config['segdata']
            config_post['segdata']['embryos'] = config_data['data_names']
            config_post['debug'] = config['debug']
            config_post['result'] = {}

            config_post['segdata']['membseg_path'] = config_data['save_folder']
            config_post['result']['postseg_folder'] = config_data['save_folder'] + 'Postseg'
            config_post['result']['nucleus_filter'] = config_test['nucleus_filter']
            config_post['result']['nucleus_as_seed'] = config_test['nucleus_as_seed']
            config_post['segdata']['max_time'] = config["para"]["max_time"]
            if not os.path.isdir(config_post['result']['postseg_folder']):
                os.makedirs(config_post['result']['postseg_folder'])

            self.post_process(process, config_post)
        return 1

    def post_process(self, process, config):

        config_segdata = config['segdata']
        membseg_path = config_segdata['membseg_path']

        # get all files under the path
        if config_segdata.get('embryos', None) is not None:
            embryo_names = config_segdata['embryos']
        else:
            embryo_names = os.listdir(membseg_path)
            embryo_names = [embryo_name for embryo_name in embryo_names if 'plc' in embryo_name.lower()]
        membseg_path = config_segdata['membseg_path']
        parameters = []
        for one_embryo in embryo_names:
            if os.path.isdir(os.path.join(config['result']['postseg_folder'], one_embryo)):
                shutil.rmtree(os.path.join(config['result']['postseg_folder'], one_embryo))
                os.makedirs(os.path.join(config['result']['postseg_folder'], one_embryo))
            else:
                os.path.join(config['result']['postseg_folder'], one_embryo)

            file_names = glob.glob(os.path.join(membseg_path, one_embryo, '*.nii.gz'))
            file_names.sort()
            for file_name in file_names[:config_segdata['max_time'] + 1]:
                parameters.append([one_embryo, file_name, config])

        self.mpPool = Pool(cpu_count() - 1)
        self.flag = True
        for idx, _ in enumerate(tqdm(self.mpPool.imap_unordered(run_post, parameters), total=len(parameters),
                                     desc="Segment binary membrane to cells")):
            #  Process name: `Segment binary membrane to cells`;  current status: `idx`; final status: `len(parameters)`;
            self.segmentation.emit('2/2 Segmenting binary membrane', idx, config_segdata['max_time'])
            if not self.flag:
                self.mpPool.close()
                return 0

        if not self.mpPool:
            self.mpPool.close()

    def run_shape_analysis(self, process, config):
        '''
        Extract the cell tree structure from the aceTree file
        :param acetree_file:  file name of the embryo acetree file
        :param max_time:  the maximum time point in the tree.
        :return :
        '''
        global max_time
        global cell_tree
        ## construct lineage tree whose nodes contain the time points that cell exist (based on nucleus).
        acetree_file = config['acetree_file']
        cell_tree, max_time = construct_celltree(acetree_file, config)
        save_file_name = os.path.join(config['stat_folder'], config['embryo_name'] + '_time_tree.txt')
        with open(save_file_name, 'wb') as f:
            pickle.dump(cell_tree, f)
        ## Parallel computing for the cell relation graph
        if not os.path.isdir(os.path.join(config["project_folder"], 'TemCellGraph')):
            os.makedirs(os.path.join(config["project_folder"], 'TemCellGraph'))

        pd_number = pd.read_csv(config["number_dictionary"], names=["name", "label"])
        number_dict = pd.Series(pd_number.label.values, index=pd_number.name).to_dict()

        # ========================================================
        #       sementing TPs in a parallel way
        # ========================================================
        file_lock = Lock()  # |-----> for change treelib files
        self.mpPool = Pool(cpu_count() - 1, initializer=init, initargs=(file_lock,))
        configs = []
        config["cell_tree"] = cell_tree
        for itime in tqdm(range(1, max_time + 1), desc="Composing configs"):
            config['time_point'] = itime
            configs.append(config.copy())
        embryo_name = config["embryo_names"][0]
        self.flag = True
        for idx, _ in enumerate(tqdm(self.mpPool.imap_unordered(cell_graph_network, configs), total=len(configs),
                                     desc="Naming {} segmentations".format(embryo_name))):
            self.analysis.emit('1/3 Naming segmentation', idx, max_time)
            if not self.flag:
                self.mpPool.close()
                return 0

        if not self.mpPool:
            self.mpPool.close()
        # ========================================================
        #       Combine previous TPs
        # ========================================================
        # ## In order to make use of parallel computing, the global vairable stat_embryo cannot be shared between different processor,
        # #  so we need to store all one-embryo reults as temporary files, which will be assembled finally. After that, these temporary
        # #  Data can be deleted.
        construct_stat_embryo(cell_tree,
                              max_time)  # initilize the shape matrix which is use to store the shape series information
        self.flag = True
        for itime in tqdm(range(1, max_time + 1), desc='assembling {} result'.format(embryo_name)):
            self.analysis.emit('2/3 Assembling results', itime, max_time)
            file_name = os.path.join(config["project_folder"], 'TemCellGraph', config['embryo_name'],
                                     config['embryo_name'] + '_T' + str(itime) + '.txt')
            with open(file_name, 'rb') as f:
                cell_graph = pickle.load(f)
                stat_embryo = assemble_result(cell_graph, itime, number_dict)
            if not self.flag:
                self.mpPool.close()
                return 0

        if not self.mpPool:
            self.mpPool.close()
        # =======================================================
        # Combine all surfaces and volumes in one single file
        # =======================================================
        # combien all volume and surface informace
        if not os.path.isdir(os.path.join(config['stat_folder'], embryo_name)):
            os.makedirs(os.path.join(config['stat_folder'], embryo_name))
        volume_lists = []
        surface_lists = []
        self.flag = True
        for t in tqdm(range(1, max_time + 1), desc="Generate surface and volume {}".format(embryo_name.split('/')[-1])):
            process.emit('3/3 Generating surface and volume', t, max_time)
            nucleus_loc_file = os.path.join(config["save_nucleus_folder"], embryo_name,
                                            os.path.basename(embryo_name) + "_" + str(t).zfill(3) + "_nucLoc" + ".csv")
            pd_loc = pd.read_csv(nucleus_loc_file)
            cell_volume_surface = pd_loc[["nucleus_name", "volume", "surface"]]
            cell_volume_surface = cell_volume_surface.set_index("nucleus_name")
            volume_lists.append(cell_volume_surface["volume"].to_frame().T.dropna(axis=1))
            surface_lists.append(cell_volume_surface["surface"].to_frame().T.dropna(axis=1))
            if not self.flag:
                self.mpPool.close()
                return 0
        if not self.mpPool:
            self.mpPool.close()
        volume_stat = pd.concat(volume_lists, keys=range(1, max_time + 1), ignore_index=True, axis=0, sort=False,
                                join="outer")
        surface_stat = pd.concat(surface_lists, keys=range(1, max_time + 1), ignore_index=True, axis=0, sort=False,
                                 join="outer")
        cell_numbers = np.count_nonzero(~np.isnan(volume_stat.to_numpy()), axis=1).astype(np.uint16).tolist()

        volume_stat = volume_stat.set_index(pd.Index(cell_numbers).astype("int64"))
        volume_stat.index.name = "Cell Number"
        surface_stat = surface_stat.set_index(pd.Index(cell_numbers).astype("int64"))
        surface_stat.index.name = "Cell Number"
        volume_stat.to_csv(
            os.path.join(config["stat_folder"], embryo_name, embryo_name.split('/')[-1] + "_volume" + '.csv'))
        surface_stat.to_csv(
            os.path.join(config["stat_folder"], embryo_name, embryo_name.split('/')[-1] + "_surface" + '.csv'))

        if config['delete_tem_file']:  # If need to delete temporary files.
            shutil.rmtree(os.path.join(config["project_folder"], 'TemCellGraph'))
        stat_embryo = stat_embryo.loc[:, ((stat_embryo != 0) & (~np.isnan(stat_embryo))).any(axis=0)]
        save_file_name_csv = os.path.join(config['stat_folder'], embryo_name, config['embryo_name'] + '_contact.csv')
        stat_embryo = stat_embryo.set_index(pd.Index(cell_numbers))
        stat_embryo.index.name = "Cell Number"
        stat_embryo.to_csv(save_file_name_csv)
        return 1

    def run_shape_analysis_nolineage(self, process, config):
        target_folder = os.path.join(config["project_folder"], "RawStack", config["embryo_names"][0], "RawMemb")
        max_time = len(os.listdir(target_folder))

        self.mpPool = Pool(cpu_count() - 1)
        configs = []
        for itime in tqdm(range(1, max_time + 1), desc="Compose configs"):
            config['time_point'] = itime
            configs.append(config.copy())

        embryo_name = config["embryo_names"][0]
        self.flag = True
        for idx, _ in enumerate(tqdm(self.mpPool.imap_unordered(analyse_seg, configs), total=len(configs),
                                     desc="Naming {} segmentations".format(embryo_name))):
            process.emit('Collecting surface and volume', idx, max_time)
            if not self.flag:
                return 0
        return 1

    def run(self):

        try:
            sin1 = self.combine_slices(self.process, self.config)
            if sin1 == 1:
                # self.signal.emit(True, 'Preprocess', "")
                if self.config["network"]["framework_name"] == "CShaper":
                    sin2 = self.test(self.process, self.config)
                elif self.config["network"]["framework_name"] == "CMap":
                    sin2 = self.test_cmap(self.process, self.config)
                if sin2 == 1:
                    # self.signal.emit(True, 'Segmentation')

                    para_config = self.config['para2']
                    # print(para_config)
                    para_config["data_folder"] = os.path.join(para_config["project_folder"], "RawStack")
                    para_config["save_nucleus_folder"] = os.path.join(para_config["project_folder"], "NucleusLoc")
                    para_config["seg_folder"] = os.path.join(para_config["project_folder"], "CellMembranePostseg")
                    para_config["stat_folder"] = os.path.join(para_config["project_folder"], "StatShape")
                    para_config["delete_tem_file"] = False
                    if not os.path.isdir(para_config['stat_folder']):
                        os.makedirs(para_config['stat_folder'])
                    # Get the size of the figure
                    example_embryo_folder = os.path.join(para_config["raw_folder"], para_config["embryo_names"][0],
                                                         "tif")
                    example_img_file = glob.glob(os.path.join(example_embryo_folder, "*.tif"))
                    raw_size = [para_config["num_slice"]] + list(np.asarray(Image.open(example_img_file[0])).shape)
                    para_config["image_size"] = [raw_size[0], raw_size[2], raw_size[1]]
                    para_config["embryo_name"] = para_config["embryo_names"][0]
                    para_config["acetree_file"] = para_config["lineage_file"]
                    if not os.path.isdir(os.path.join(para_config['save_nucleus_folder'], para_config['embryo_name'])):
                        os.makedirs(os.path.join(para_config['save_nucleus_folder'], para_config['embryo_name']))
                    else:
                        shutil.rmtree(os.path.join(para_config['save_nucleus_folder'], para_config['embryo_name']))
                        os.makedirs(os.path.join(para_config['save_nucleus_folder'], para_config['embryo_name']))
                    if para_config["acetree_file"] != '':
                        if para_config['lineage_file'] == '':
                            sin3 = self.run_shape_analysis_nolineage(self.process, para_config)
                        else:
                            sin3 = self.run_shape_analysis(self.process, para_config)
                        if sin3 == 1:
                            if not self.mpPool:
                                self.mpPool.close()
                            self.signal.emit(True, "Analysis", 'All process completed')
            # if USE_GPU:
            #     device = cuda.get_current_device()
            #     device.reset()
            #     cuda.close()

        except Exception:
            if not self.mpPool:
                self.mpPool.close()
            # if USE_GPU:
            #     device = cuda.get_current_device()
            #     device.reset()
            #     cuda.close()
            self.signal.emit(False, 'All programs', traceback.format_exc())


class TrainThread(QThread):
    signal = pyqtSignal(bool, str, str)
    process = pyqtSignal(str, int, int)

    def __init__(self, config={}):
        self.config = config
        self.flag = None
        super(TrainThread, self).__init__()

    def __del__(self):
        self.wait()

    def threadflag(self, flag):
        self.flag = flag

    def train(self, process, config):

        # =============================================================
        #               1, Load configuration parameters
        # =============================================================
        class_num = 16

        # ==============================================================
        #               2, Construct computation graph
        # ==============================================================
        tf.compat.v1.reset_default_graph()
        with tf.name_scope('model_builder'):
            w_regularizer = regularizers.L2(1e-7)
            b_regularizer = regularizers.L2(1e-7)
            net = DMapNetCompiled(input_size=[24, 128, 128, 1],
                                  num_classes=class_num,
                                  kernel_regularizer=w_regularizer,
                                  bias_regularizer=b_regularizer,
                                  activation="relu")

        # ==============================================================
        #               3, Data loader
        # ==============================================================
        process.emit('Loading data', -1, 2)
        dataloader = DataGene(dict(data_root=config["data"],
                                   data_names=config["embryo_names"],
                                   with_ground_truth=True,
                                   batch_size=config["batch"],
                                   data_shape=[24, 128, 128, 1],
                                   label_shape=[16, 128, 128, 1],
                                   label_edt_transform=True,
                                   valid_edt_width=30,
                                   label_edt_discrete=True,
                                   edt_discrete_num=16))
        epoches = config["maximal_iteration"] // len(dataloader.data)
        opt = tf.keras.optimizers.Adam(learning_rate=5e-4)
        # net.compile(optimizer=opt, loss=weighted_one_hot_loss(class_num))
        loss = weighted_one_hot_loss(class_num)
        # progressbar = ProgressBar(progress=process, total_epoch=epoches)
        self.flag = True
        process.emit('Training model', -1, epoches)
        for epoch in tqdm(range(epoches), desc="Training model"):
            for step, (x_data, y_data) in enumerate(dataloader):
                with tf.GradientTape() as tape:
                    pred = net(x_data, training=True)
                    loss_value = loss(y_data, pred)
            grads = tape.gradient(loss_value, net.trainable_weights)
            opt.apply_gradients(zip(grads, net.trainable_weights))
            process.emit('Training model', epoch, epoches)
            if not self.flag:
                del dataloader, net
                gc.collect()
                return 0
        # results = net.fit(dataloader, epochs=epoches, shuffle=True, workers=8, callbacks=[progressbar])

        # ==============================================================
        #               3, Start train
        # ==============================================================
        net.save_weights(os.path.join(config["save_folder"], "{}_{}.ckpt".format(config["name"], str(config["maximal_iteration"]).zfill(5))))

        del dataloader, net
        gc.collect()
        return 1

    def train_cmap(self, process, config):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = EDTDMFNet(in_channels=1, n_first=32, conv_channels=64, groups=16, norm="in", out_class=1)
        model = model.to(device)

        optimizer = Adam(model.parameters(), lr=0.005, weight_decay=0.00001, amsgrad=True)
        train_set = Memb3DDataset(config["data"], membrane_names=config["embryo_names"], for_train=True,
                                return_target=True,
                                transforms="Compose([ContourEDT(9),RandomIntensityChange([0.1, 0.1]),RandCrop((128,128,128)),RandomFlip(0),NumpyType((np.float32, np.float32, np.float32, np.float32))])",
                                suffix="*.nii.gz")
        num_iters = (len(train_set) * 50) // 2
        train_sampler = CycleSampler(len(train_set), num_iters * 2)
        train_loader = DataLoader(
            dataset=train_set,
            batch_size=2,
            # collate_fn=train_set.collate,
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True,
            worker_init_fn=init_fn
        )

        torch.set_grad_enabled(True)
        enum_batches = len(train_set) / 2  # number of epoches in each iteration
        self.flag = True
        process.emit('Training model', -1, 775)
        num_iter = 0
        for i, data in enumerate(tqdm(train_loader, desc="Training")):
            #  record process
            elapsed_bsize = int(i / enum_batches) + 1
            epoch = int((i + 1) / enum_batches)
            num_iter += 1

            #  adjust learning super parameters
            adjust_learning_rate(optimizer, epoch, num_iters, 0.005)

            #  go through the network
            data = [t.to(device) for t in data]  # Set non_blocking for multiple GPUs
            raw, target_dis = data
            predict_dis = model(raw)
            loss = mse_loss(predict_dis, target_dis)
            #  backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            process.emit("Training model", num_iter, 775)
            if not self.flag:
                return 0

        file_name = os.path.join(config["save_folder"], "{}_{}.pth".format(config["name"], config["model"]))
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        torch.save(dict(
            iter=i,
            state_dict=model.state_dict(),
            optim_dict=optimizer.state_dict()
        ), file_name)

        return 1


    def run(self):
        try:
            sin = 0
            if self.config["train"]["model"] == "CShaper":
                sin = self.train(self.process, self.config["train"])
            elif self.config["train"]["model"] == "CMap":
                sin = self.train_cmap(self.process, self.config["train"])
            if sin == 1:
                self.signal.emit(True, 'Train', 'Train Completed!')
            # if USE_GPU:
            #     device = cuda.get_current_device()
            #     print("**** Error in getting device")
            #     device.reset()  # TODO: GPU cannot be used again after releasing
            #     print("**** Error in resetting device")
            #     cuda.close()
        except Exception:
            self.signal.emit(False, 'train', traceback.format_exc())
            # if USE_GPU:
            #     device = cuda.get_current_device()
            #     device.reset()
            #     cuda.close()