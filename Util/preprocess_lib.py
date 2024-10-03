import os
import glob
import warnings
import shutil
warnings.filterwarnings("ignore")
import numpy as np
from PIL import Image
import nibabel as nib
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
from skimage.transform import resize

# import user defined library
from Util.parse_config import parse_config

def combine_slices(process, config):
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
    mpPool = mp.Pool(mp.cpu_count() - 1)

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

        for idx, _ in enumerate(tqdm(mpPool.imap_unordered(stack_nuc_slices, configs), total=len(configs),
                                     desc="1/3 Stack nucleus of {}".format(embryo_name))):
            # TODO: Process Name: `1/3 Stack nucleus`; Current status: `idx`; Final status: max_time
            process.emit('1/3 Stack nucleus', idx, max_time)
            # pass
            # stack_nuc_slices(raw_folder=origin_folder, save_folder=target_folder, embryo_name=embryo_name, tp=tp,
            #                  out_size=out_size, num_slice=num_slice, res=out_res)

        # save membrane
        origin_files = glob.glob(os.path.join(raw_folder, embryo_name, "tifR", "*.tif"))
        origin_files = sorted(origin_files)
        target_folder = os.path.join(stack_folder, embryo_name, "RawMemb")
        if not os.path.isdir(target_folder):
            os.makedirs(target_folder)

        configs = []
        for tp in range(1, max_time + 1):
            configs.append((origin_files, target_folder, embryo_name, tp, out_size, num_slice, out_res))
        for idx, _ in enumerate(tqdm(mpPool.imap_unordered(stack_memb_slices, configs), total=len(configs),
                                     desc="2/3 Stack membrane of {}".format(embryo_name))):
            # TODO: Process Name: `2/3 Stack membrane`; Current status: `idx`; Final status: max_time
            process.emit('2/3 Stack membrane', idx, max_time)
        # for tp in range(1, max_time+1):
        #     stack_memb_slices(raw_folder=origin_folder, save_folder=target_folder, embryo_name=embryo_name, tp=tp,
        #                     out_size=out_size, num_slice=num_slice, res=out_res)

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
            for idx, _ in enumerate(tqdm(mpPool.imap_unordered(save_nuc_seg, configs), total=len(configs),
                                         desc="3/3 Construct nucleus location of {}".format(embryo_name))):
                # TODO: Process Name: `3/3 Construct nucleus location`; Current status: `idx`; Final status: max_time
                process.emit('3/3 Construct nucleus location', idx, max_time)
            # for tp in range(1, max_time+1):
            #     save_nuc_seg(embryo_name=embryo_name,
            #                  name_dict=name_dict,
            #                  pd_lineage=pd_lineage,
            #                  tp=tp,
            #                  raw_size=raw_size,
            #                  out_size=out_size,
            #                  out_res=out_res,
            #                  dif_res=xy_res/z_res,
            #                  save_folder=target_folder)
            shutil.copy(lineage_file, os.path.join(stack_folder, embryo_name))

# ============================================
# save raw nucleus stack
# ============================================
def stack_nuc_slices(para):
    [origin_files, save_folder, embryo_name, tp, out_size, num_slice, res] = para

    out_stack = []
    save_file_name = "{}_{}_rawNuc.nii.gz".format(embryo_name, str(tp).zfill(3))
    for idx in range((tp-1)*num_slice, tp*num_slice):
        raw_file_name = origin_files[idx]
        img = np.asanyarray(Image.open(raw_file_name))
        out_stack.insert(0, img)
    img_stack = np.transpose(np.stack(out_stack), axes=(1, 2, 0))
    img_stack = resize(image=img_stack, output_shape=out_size, preserve_range=True, order=1).astype(np.uint8)
    nib_stack = nib.Nifti1Image(img_stack, np.eye(4))
    nib_stack.header.set_xyzt_units(xyz=3, t=8)
    nib_stack.header["pixdim"] = [1.0, res[0], res[1], res[2], 0., 0., 0., 0.]
    nib.save(nib_stack, os.path.join(save_folder, save_file_name))

# ============================================
# save raw membrane stack
# ============================================

def stack_memb_slices(para):
    [origin_files, save_folder, embryo_name, tp, out_size, num_slice, res] = para

    out_stack = []
    save_file_name = "{}_{}_rawMemb.nii.gz".format(embryo_name, str(tp).zfill(3))
    for idx in range((tp-1)*num_slice, tp*num_slice):
        raw_file_name = origin_files[idx]

        img = np.asanyarray(Image.open(raw_file_name))
        out_stack.insert(0, img)
    img_stack = np.transpose(np.stack(out_stack), axes=(1, 2, 0))
    img_stack = resize(image=img_stack, output_shape=out_size, preserve_range=True, order=1).astype(np.uint8)
    nib_stack = nib.Nifti1Image(img_stack, np.eye(4))
    nib_stack.header.set_xyzt_units(xyz=3, t=8)
    nib_stack.header["pixdim"] = [1.0, res[0], res[1], res[2], 0., 0., 0., 0.]
    nib.save(nib_stack, os.path.join(save_folder, save_file_name))

# =============================================
# save nucleus segmentation
# =============================================
def save_nuc_seg(para):
    [embryo_name, name_dict, pd_lineage, tp, raw_size, out_size, out_res, dif_res, save_folder] = para

    zoom_ratio = [y / x for x, y in zip(raw_size, out_size)]
    tp_lineage = pd_lineage[pd_lineage["time"] == tp]
    tp_lineage.loc[:, "x"] = (tp_lineage["x"] * zoom_ratio[0]).astype(np.int16)
    tp_lineage.loc[:, "y"] = (np.floor(tp_lineage["y"] * zoom_ratio[1])).astype(np.int16)
    tp_lineage.loc[:, "z"] = (out_size[2] - np.floor(tp_lineage["z"] * (zoom_ratio[2] / dif_res))).astype(np.int16)

    # !!!! x <--> y
    nuc_dict = dict(
        zip(tp_lineage["cell"], zip(tp_lineage["y"].values, tp_lineage["x"].values, tp_lineage["z"].values)))
    # print(name_dict)
    labels = [name_dict[name] for name in list(nuc_dict.keys())]
    locs = list(nuc_dict.values())
    out_seg = np.zeros(out_size, dtype=np.uint16)
    out_seg[tuple(zip(*locs))] = labels

    save_file_name = "_".join([embryo_name, str(tp).zfill(3), "segNuc.nii.gz"])
    nib_stack = nib.Nifti1Image(out_seg, np.eye(4))
    nib_stack.header.set_xyzt_units(xyz=3, t=8)
    nib_stack.header["pixdim"] = [1.0, out_res[1], out_res[0], out_res[2], 0., 0., 0., 0.]
    nib.save(nib_stack, os.path.join(save_folder, save_file_name))


if __name__ == "__main__":
    os.chdir("../")
    config = parse_config("./ConfigMemb/test_edt_discrete.txt")
    config = config["para"]
    combine_slices(1,config)