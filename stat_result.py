import os
import glob
import multiprocessing as mp
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import ndimage
from scipy.ndimage.morphology import binary_erosion, binary_dilation

# import user defined library
from Util.data_process import load_3d_volume_as_array
from shape_analysis import get_contact_area, get_surface_area


def get_contact(volume):
    contact_pairs, contact_areas = get_contact_area(volume)

    return contact_pairs, contact_areas


def save_pd(pd, save_file):
    if not os.path.isdir(os.path.dirname(save_file)):
        os.makedirs(os.path.dirname(save_file))
    pd.to_csv(save_file)


# ==========================================
# codes with nucleus
# ==========================================
def stat_embryo(embryo_name):
    preresult_folder = "./ResultCell/BinaryMembPostseg"
    postresult_folder = "./ResultCell/BinaryMembPostseg"

    preresult_files = glob.glob(os.path.join(preresult_folder, embryo_name+"Cavity", "*.nii.gz"))
    preresult_files.sort()
    postresult_files = glob.glob(os.path.join(postresult_folder, embryo_name+"LabelUnified", "*.nii.gz"))
    postresult_files.sort()
    for index, postresult_file in enumerate(tqdm(postresult_files, desc="Process {}".format(embryo_name))):
        base_name = "_".join(os.path.basename(postresult_file).split("_")[:-1])

        preresult_file = os.path.join(preresult_folder, embryo_name+"Cavity", base_name + "_segCavity.nii.gz")
        postresult = load_3d_volume_as_array(postresult_file)

        if os.path.isfile(preresult_file):
            presult = load_3d_volume_as_array(preresult_file)
            # errors = np.logical_and(presult != 0, ~(postresult != 0)).astype(np.uint8)
            extra_label_volume = ndimage.label(presult)[0]
            extra_labels = np.unique(extra_label_volume).tolist()
            extra_labels.remove(0)

            # combine labels
            extra_label_init = 10000
            for idx, extra_label in enumerate(extra_labels):
                postresult[extra_label_volume == extra_label] = extra_label_init + idx

        # get surface, volume
        labels = []
        volumes = []
        surfaces = []
        postlabels = np.unique(postresult).tolist()
        postlabels.remove(0)
        for label in postlabels:
            surface = get_surface_area(postresult == label)

            volume = (postresult == label).sum()

            labels.append(label)
            volumes.append(volume)
            surfaces.append(surface)
        contact_pairs, contact_areas = get_contact(postresult)
        contact_surfaces = {contact_pairs[i][0]: {contact_pairs[i][1]: contact_areas[i]} for i in range(len(contact_areas))}

        # save to csv file
        save_volme_file = os.path.join("./Tem/Stat", embryo_name, base_name + "_surface_volume.csv")
        save_contact_file = os.path.join("./Tem/Stat", embryo_name, base_name + "_contact.csv")

        pd_surface_volume = pd.DataFrame.from_dict({"Label": labels, "Surface": surfaces, "Volume": volumes})
        pd_surface_volume.set_index("Label", inplace=True)
        save_pd(pd=pd_surface_volume, save_file=save_volme_file)
        pd_contact = pd.DataFrame.from_dict({i: contact_surfaces[i] for i in contact_surfaces.keys()}, orient="index")
        save_pd(pd=pd_contact, save_file=save_contact_file)


# ==========================================
# codes without nucleus
# ==========================================
def stat_embryo_no_nucleus(embryo_name):
    preresult_folder = "ResultCell/BinaryMembPostseg"

    preresult_files = glob.glob(os.path.join(preresult_folder, embryo_name, "*.nii.gz"))
    preresult_files.sort()
    for index, preresult_file in enumerate(tqdm(preresult_files, desc="Process {}".format(embryo_name))):
        base_name = "_".join(os.path.basename(preresult_file).split("_")[:-1])

        preresult = load_3d_volume_as_array(preresult_file)

        # get surface, volume
        labels = []
        volumes = []
        surfaces = []
        postlabels = np.unique(preresult).tolist()
        postlabels.remove(0)


        for label in postlabels:
            surface = get_surface_area(preresult == label)

            volume = (preresult == label).sum()

            labels.append(label)
            volumes.append(volume)
            surfaces.append(surface)
        contact_surfaces = {}
        if len(postlabels) > 1:
            contact_pairs, contact_areas = get_contact(preresult)
            contact_surfaces = {contact_pairs[i][0]: {contact_pairs[i][1]: contact_areas[i]} for i in range(len(contact_areas))}

        # save to csv file
        save_volme_file = os.path.join("./Tem/Stat", embryo_name, base_name + "_surface_volume.csv")
        save_contact_file = os.path.join("./Tem/Stat", embryo_name, base_name + "_contact.csv")

        pd_surface_volume = pd.DataFrame.from_dict({"Label": labels, "Surface": surfaces, "Volume": volumes})
        pd_surface_volume.set_index("Label", inplace=True)
        save_pd(pd=pd_surface_volume, save_file=save_volme_file)
        pd_contact = pd.DataFrame.from_dict({i: contact_surfaces[i] for i in contact_surfaces.keys()}, orient="index")
        save_pd(pd=pd_contact, save_file=save_contact_file)


if __name__ == "__main__":
    # embryo_names = ["200314plc1p1", "181210plc1p3", "200314plc1p2"] + \
    #     "200309plc1p2, 200309plc1p3, 200310plc1p2, 200311plc1p1, 200315plc1p2, 200315plc1p3, 200316plc1p1, 200316plc1p2".split(",") + \
    #     "200309plc1p1, 200312plc1p2".split(",") + \
    #     "200311plc1p2, 200311plc1p3, 200312plc1p1, 200312plc1p3, 200314plc1p3, 200315plc1p1, 200316plc1p3, 181210plc1p1".split(",") + \
    #     "181210plc1p2, 170704plc1p1".split(",") + \
    #     ["200315plc1p1", "200311plc1p2", "200310plc1p2"]
    # embryo_names = [embryo_name.replace(" ", "") for embryo_name in embryo_names]

    # embryo_names = ["200315plc1p1", "200311plc1p2", "200310plc1p2"]
    # embryo_names = ["200113plc1p2"]
    # embryo_names = ["200710hmr1plc1p1", "200710hmr1plc1p2", "200710hmr1plc1p3"]

    embryo_names = ["200113plc1p2", "181210plc1p2", "200310plc1p2", "170704plc1p1", "181210plc1p1",
                    "200316plc1p3", "200312plc1p2", "200311plc1p3", "200309plc1p3", "200315plc1p2",
                    "200316plc1p2", "200311plc1p1", "200312plc1p3", "200316plc1p1", "200314plc1p3",
                    "200312plc1p1", "200315plc1p3"]
    # stat_embryo_no_nucleus(embryo_names[0])
    with mp.Pool(processes=16) as p:
        p.map(stat_embryo, embryo_names)







