#  import dependency library
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

# import user defined
from Util.data_utils import get_all_stack, pkload
from Util.data_process import itensity_normalize_one_volume, load_3d_volume_as_array
from Util.augmentations import contour_distance, contour_distance_outside_negative
from Util.transforms import Compose, RandCrop, RandomFlip, NumpyType, RandomRotation, Pad, Resize, ContourEDT, RandomIntensityChange


#=======================================
#  Import membrane datasets
#=======================================
#   data format: dict([raw_memb, raw_nuc, seg_nuc, 'seg_memb, seg_cell'])
class Memb3DDataset(Dataset):
    def __init__(self, root="dataset/train", membrane_names=None, for_train=True, return_target=True,
                 transforms=None, suffix="*.pkl", max_times=None):
        if membrane_names is None:
            membrane_names = [name for name in os.listdir(root) if os.path.isdir(os.path.join(root, name))]
        self.paths = get_all_stack(root, membrane_names, suffix=suffix, max_times=max_times)
        self.seg_memb_paths = get_all_stack(root, membrane_names, suffix=suffix, max_times=max_times, sub_dir="SegMemb")
        self.names = [os.path.basename(path).split(".")[0] for path in self.paths]
        self.for_train = for_train
        self.return_target = return_target
        self.transforms = eval(transforms or "Identity()")  #
        self.size = self.get_size()

    def __getitem__(self, item):
        stack_name = self.names[item]
        raw_memb = load_3d_volume_as_array(self.paths[item])
        if self.return_target:
            seg_memb = load_3d_volume_as_array(self.seg_memb_paths[item])
            target_distance = contour_distance(seg_memb, d_threshold=15)
            raw, seg_dis = self.transforms([raw_memb, target_distance])
            raw, seg_dis = self.volume2tensor([raw, seg_dis], dim_order=[0, 1, 2])
            return raw, seg_dis
        else:
            raw_memb = self.transforms(raw_memb)
            return raw_memb[None], self.paths[item]
    def volume2tensor(self, volumes0, dim_order = None):
        volumes = volumes0 if isinstance(volumes0, list) else [volumes0]
        outputs = []
        for volume in volumes:
            volume = volume.transpose(dim_order)[np.newaxis, ...]
            volume = np.ascontiguousarray(volume)
            volume = torch.from_numpy(volume)
            outputs.append(volume)

        return outputs if isinstance(volumes0, list) else outputs[0]

    def get_size(self):
        # print(self.paths)
        raw_memb_shape = load_3d_volume_as_array(self.paths[0]).shape

        return raw_memb_shape

    def __len__(self):
        return len(self.names)
