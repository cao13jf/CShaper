from __future__ import absolute_import, print_function

from tqdm import tqdm
from Util.data_process import *


class DataLoader():
    def __init__(self, config):
        """
        Initialize the calss instance
        inputs:
            config: a dictionary representing parameters
        """
        self.config = config
        self.data_root = config['data_root'] if type(config['data_root']) is list else [
            config['data_root']]  # Save as list
        self.intensity_normalize = config.get('intensity_normalize', True)
        self.with_ground_truth = config.get('with_ground_truth', False)
        self.label_convert_source = config.get('label_convert_source', None)
        self.label_convert_target = config.get('label_convert_target', None)
        self.label_postfix = config.get('label_postfix', 'seg')
        self.file_postfix = config.get('file_postfix', 'nii')
        self.data_names = config.get('data_names', None)
        self.data_num = config.get('data_num', None)
        self.data_shape = config.get('data_shape', None)
        self.crop = config.get('roi_crop', False)
        self.add_boundary_layers = config.get('add_boundary_layers', False)
        ##  Data augmentation
        self.data_resize = config.get('data_resize', None)
        self.with_translate = config.get('with_translate', False)
        self.with_scale = config.get('with_scale', False)
        self.with_rotation = config.get('with_rotation', False)
        self.with_flip = config.get('with_flip', False)
        # Get EDT pre-processing information
        self.label_edt_transform = config.get('label_edt_transform', False)
        self.valid_edt_width = config.get('valid_edt_value', 30)
        self.label_edt_discrete = config.get('label_edt_discrete', False)
        self.edt_discrete_num = config.get('edt_discrete_num', 16)

        if (self.label_convert_source and self.label_convert_target):
            assert (len(self.label_convert_source) == len(self.label_convert_target))

    def __get_embryo_names(self):
        """
        get the list of patient names, if self.data_names id not None, then load patient
        names from that file, otherwise search all the names automatically in data_root
        """
        # use pre-defined patient names
        embryo_names = os.listdir(self.data_root[0])
        embryo_names = [name for name in embryo_names if 'plc' in name.lower()]
        if (self.data_names is not None):
            embryo_names = self.data_names

        return embryo_names

    def __load_one_volume(self, volume_dir, one_volume_name):
        # for membrane Data
        volume_name = one_volume_name
        assert (volume_name is not None)
        volume = load_3d_volume_as_array(os.path.join(volume_dir, volume_name))
        return volume, volume_name

    def load_data(self):
        """
        load all the training/testing Data
        """
        self.embryo_names = self.__get_embryo_names()
        assert (len(self.embryo_names) > 0)  # Exit with empty name list
        ImageNames = []
        embryoNames = []
        X = []
        W = []
        Y = []
        bbox = []
        in_size = []
        data_num = self.data_num if (self.data_num is not None) else len(self.embryo_names)
        for one_embryo_name in self.embryo_names:
            raw_path = os.path.join(self.data_root[0], one_embryo_name, 'rawMemb')
            mask_path = os.path.join(self.data_root[0], one_embryo_name, 'segMemb')
            volumes_lists = os.listdir(raw_path)
            for one_volume_name in tqdm(volumes_lists, desc='Loading Data in ' + raw_path):
                # load rawMemb volume
                volume, volume_name = self.__load_one_volume(raw_path, one_volume_name)
                ## add boundary layers for better boundary discrimination
                if (self.add_boundary_layers):
                    append_layers = np.zeros((10, volume.shape[1], volume.shape[2]), dtype=np.float32)
                    volume = np.append(append_layers, volume, axis=0)
                    volume = np.append(volume, append_layers, axis=0)

                bbmin, bbmax = [0, 0, 0], [y - 1 for y in volume.shape]
                if (self.crop):
                    margin = 5
                    bbmin, bbmax = get_ND_bounding_box(volume, margin)
                    volume = crop_ND_volume_with_bounding_box(volume, bbmin, bbmax)
                bbox.append([bbmin, bbmax])  # Box size
                if (self.data_resize):  # Whether resize the volume
                    volume = resize_ND_volume_to_given_shape(volume, self.data_resize, 1)
                if (self.intensity_normalize):
                    volume = itensity_normalize_one_volume(volume)
                volume_size = volume.shape
                weight = np.asarray(volume > 0, np.float32)

                X.append(volume)
                W.append(weight)  # What is the weight for mod 'flair' ???
                embryoNames.append(one_embryo_name)
                ImageNames.append(volume_name)
                in_size.append(volume_size)  # Volume size

                if (self.with_ground_truth):  # Label includes 5 labels [0, 1, 2, 3, 4]
                    mask_name = "_".join(volume_name.split(".")[0].split("_")[0:2] + ["segMemb.nii.gz"])
                    label, _ = self.__load_one_volume(mask_path, mask_name)
                    if (self.add_boundary_layers):
                        append_layers = np.zeros((10, volume.shape[1], volume.shape[2]), dtype=np.float32)
                        label = np.append(append_layers, label, axis=0)
                        label = np.append(label, append_layers, axis=0)

                    if (self.crop):
                        label = crop_ND_volume_with_bounding_box(label, bbmin, bbmax)
                    if (self.data_resize):
                        label = resize_ND_thin_mask_to_given_shape(label, self.data_resize)
                    # check whether implement EDT on label
                    if (self.label_edt_transform):
                        if (self.label_edt_discrete):
                            label = binary_to_EDT_3D(label, self.valid_edt_width, self.edt_discrete_num).astype(
                                np.uint8)
                        else:
                            label = binary_to_EDT_3D(label, self.valid_edt_width).astype(np.uint8)
                    Y.append(label)
        self.image_names = ImageNames
        self.embryo_names = embryoNames
        self.data = X
        self.weight = W
        self.label = Y
        self.bbox = bbox
        self.in_size = in_size

    def get_subimage_batch(self):
        """
        sample a batch of image patches for segmentation. Only used for training
        """
        flag = False
        while (flag == False):
            batch = self.__get_one_batch()
            labels = batch['labels']
            if (labels.sum() > 0):
                flag = True
        return batch

    def __get_one_batch(self):  # internal method without being overwrited
        """
        get a batch from training Data
        """
        batch_size = self.config['batch_size']
        data_shape = self.config['data_shape']
        label_shape = self.config['label_shape']
        down_sample_rate = self.config.get('down_sample_rate', 1.0)
        data_slice_number = data_shape[0]  # slice number locates at first
        label_slice_number = label_shape[0]  # Why not the same ?
        batch_sample_model = self.config.get('batch_sample_model', ('full', 'valid', 'valid'))
        slice_direction = self.config.get('slice_direction', 'axial')  # axial, sagittal, coronal or random
        train_with_roi_patch = self.config.get('train_with_roi_patch', False)
        keep_roi_outside = self.config.get('keep_roi_outside', False)
        if (train_with_roi_patch):
            label_roi_mask = self.config['label_roi_mask']
            roi_patch_margin = self.config['roi_patch_margin']

        # return batch size: [batch_size, slice_num, slice_h, slice_w, moda_chnl]
        data_batch = []
        weight_batch = []
        label_batch = []
        if (slice_direction == 'random'):  # random slice order
            directions = ['axial', 'sagittal', 'coronal']
            idx = random.randint(0, 2)
            slice_direction = directions[idx]
        for i in range(batch_size):
            self.image_id = random.randint(0, len(self.data) - 1)  # random chose dataset
            data_volume = self.data[self.image_id]
            weight_volume = self.weight[self.image_id]
            boundingbox = None
            if (self.with_ground_truth):
                label_volume = self.label[self.image_id]
                ## Data augmentation
                [data_volume, weight_volume, label_volume] = self.augment_data(
                    [data_volume, weight_volume, label_volume], issegs=[False, True, True])
                if (train_with_roi_patch):
                    mask_volume = np.zeros_like(label_volume)  # Just one label volume but saved as list
                    [d_idxes, h_idxes, w_idxes] = np.nonzero(mask_volume)
                    [D, H, W] = label_volume.shape  # Get bounding box of label
                    mind = max(d_idxes.min() - roi_patch_margin, 0)
                    maxd = min(d_idxes.max() + roi_patch_margin, D)
                    minh = max(h_idxes.min() - roi_patch_margin, 0)
                    maxh = min(h_idxes.max() + roi_patch_margin, H)
                    minw = max(w_idxes.min() - roi_patch_margin, 0)
                    maxw = min(w_idxes.max() + roi_patch_margin, W)
                    if (keep_roi_outside):  # Whether crop all corresponding volumes
                        boundingbox = [mind, maxd, minh, maxh, minw, maxw]
                    else:
                        data_volume = data_volume[np.ix_(range(mind, maxd),
                                                         range(minh, maxh),
                                                         range(minw, maxw))]
                        weight_volume = weight_volume[np.ix_(range(mind, maxd),
                                                             range(minh, maxh),
                                                             range(minw, maxh))]
                        label_volume = label_volume[np.ix_(range(mind, maxd),
                                                           range(minh, maxh),
                                                           range(minw, maxw))]
            else:
                [data_volume, weight_volume] = self.augment_data([data_volume, weight_volume], issegs=[False, True])

                # if(self.label_convert_source and self.label_convert_target):
                #   label_volume[0] = convert_label(label_volume[0], self.label_convert_source, self.label_convert_target)

            transposed_volume = transpose_volumes(data_volume, slice_direction)  # Transpose the direction of volume
            volume_shape = transposed_volume.shape
            sub_data_shape = [data_slice_number, data_shape[1], data_shape[2]]
            sub_label_shape = [label_slice_number, label_shape[1], label_shape[2]]
            center_point = get_random_roi_sampling_center(volume_shape, sub_label_shape, batch_sample_model,
                                                          boundingbox)
            sub_data = extract_roi_from_volume(transposed_volume, center_point, sub_data_shape)
            # if(rotation):

            transposed_weight = transpose_volumes(weight_volume, slice_direction)
            sub_weight = extract_roi_from_volume(transposed_weight,
                                                 center_point,
                                                 sub_label_shape,
                                                 fill='zero')
            if (self.with_ground_truth):
                tranposed_label = transpose_volumes(label_volume, slice_direction)
                sub_label = extract_roi_from_volume(tranposed_label,
                                                    center_point,
                                                    sub_label_shape,
                                                    fill='zero')

            weight_batch.append([sub_weight])
            data_batch.append([sub_data])
            if (self.with_ground_truth):
                label_batch.append([sub_label])

        data_batch = np.asarray(data_batch, np.float32)
        weight_batch = np.asarray(weight_batch, np.float32)
        label_batch = np.asarray(label_batch, np.float)
        batch = {}

        batch['images'] = np.transpose(data_batch, [0, 2, 3, 4, 1])
        batch['weights'] = np.transpose(weight_batch, [0, 2, 3, 4, 1])
        batch['labels'] = np.transpose(label_batch, [0, 2, 3, 4, 1])

        return batch

    def get_total_image_number(self):
        """
        get the toal number of images
        """
        return len(self.data)

    def get_image_data_with_name(self, i):
        """
        Used for testing, get one image Data and patient name
        """
        return [self.data[i], self.weight[i], self.image_names[i], self.embryo_names[i], self.bbox[i], self.in_size[i]]

    '''
    functions for augmentation
    '''

    def augment_data(self, imgs, issegs=[False]):
        random_nums = [random.random() for i in range(4)]  # control  transform ratio
        out_imgs = []
        for i, img in enumerate(imgs):
            if (self.with_translate) and (random_nums[0] > 0.5):
                offset = [random.randint(-10, 10), random.randint(-10, 10), random.randint(-5, 5)]
                img = self.translate_it(img, offset, isseg=issegs[i])

            if (self.with_scale) and (random_nums[1] > 0.5):  # with random probability
                factor = [random.uniform(0.9, 1.1), random.uniform(0.9, 1.1), random.uniform(0.9, 1.1)]
                img = self.scale_it(img, factor, isseg=issegs[i])

            if (self.with_rotation) and (random_nums[2] > 0.5):
                theta = random.uniform(30, -30)
                axes = (1, 0)
                img = self.rotate_it(img, theta, axes, isseg=issegs[i])

            if (self.with_flip) and (random_nums[3] > 0.5):
                axes = random.choice([0, 1, 2])
                img = self.flip_it(img, axes)
            out_imgs.append(img)

        return out_imgs

    def translate_it(self, img, offset=[10, 10, 2], isseg=False, mode='nearest'):
        order = 0 if isseg == True else 5

        return ndimage.interpolation.shift(img, (int(offset[0]), int(offset[1]), int(offset[2])), order=order,
                                           mode=mode)

    def scale_it(self, img, factor=[0.8, 0.8, 0.9], isseg=False, mode='nearest'):
        order = 0 if isseg == True else 3

        height, width, depth = img.shape
        zheight = int(np.round(factor * height))
        zwidth = int(np.round(factor * width))
        zdepth = depth

        if factor < 1.0:
            newimg = np.zeros_like(img)
            row = (height - zheight) // 2
            col = (width - zwidth) // 2
            layer = (depth - zdepth) // 2
            newimg[row:row + zheight, col:col +
                                          zwidth, layer:layer + zdepth] = ndimage.interpolation.zoom(img, (
            float(factor[0]), float(factor[1]), float(factor[2])),
                                                                                                     order=order,
                                                                                                     mode=mode)[
                                                                          0:zheight, 0:zwidth, 0:zdepth]
            return newimg

        elif factor > 1.0:
            row = (zheight - height) // 2
            col = (zwidth - width) // 2
            layer = (zdepth - depth) // 2

            newimg = ndimage.interpolation.zoom(img[row:row + zheight, col:col + zwidth, layer:layer + zdepth],
                                                (float(factor[0]), float(factor[1]), float(factor[2])), order=order,
                                                mode=mode)
            extrah = (newimg.shape[0] - height) // 2
            extraw = (newimg.shape[1] - width) // 2
            extrad = (newimg.shape[2] - depth) // 2
            newimg = newimg[extrah:extrah + height, extraw:extraw + width, extrad:extrad + depth]

            return newimg

        else:
            return img

    def rotate_it(self, img, theta, axes=(1, 0), isseg=False, mode='constant'):
        order = 0 if isseg == True else 5

        return ndimage.rotate(img, float(theta), axes=axes, reshape=False, order=order, mode=mode)

    def flip_it(self, img, axes=0):

        return np.flip(img, axes)

