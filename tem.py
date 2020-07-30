
import os
import glob
import numpy as np
from tqdm import tqdm

from Util.data_process import load_3d_volume_as_array

root_folder = "./ResultCell/BinaryMembPostseg/200315plc1p1"
files = glob.glob(os.path.join(root_folder, "*.nii.gz"))
files.sort()
nums = []
for i, file in enumerate(tqdm(files)):
    volume = load_3d_volume_as_array(file)
    labels = np.unique(volume).tolist()
    labels.remove(0)

    nums.append(len(labels))

zeros = [i+1 for i, num in enumerate(nums) if num == 0]
print(zeros)