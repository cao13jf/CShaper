import os
import glob
import numpy as np
from tqdm import tqdm
import nibabel as nib
from Util.data_process import save_array_as_nifty_volume


def save_nii(img, nii_name):
    nii_folder = os.path.dirname(nii_name)
    if not os.path.isdir(nii_folder):
        os.makedirs(nii_folder)
    img = nib.Nifti1Image(img, np.eye(4))
    nib.save(img, nii_name)

root_folder = "ResultCell/BothWithRandomnetPostseg/181210plc1p2"
all_files = []
for folder in ["rawMemb"]:
    files = glob.glob(os.path.join(root_folder, "*.nii.gz"))
    for file in tqdm(files):
        aa0 = nib.load(file).get_fdata()
        aa = np.flip(aa0, axis=2)

        # tp_str = os.path.basename(file).split(".")[0][5:]
        save_nii(aa, os.path.join(root_folder, os.path.basename(file)))