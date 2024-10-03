
# import dependency library
import sys
import shutil
import warnings
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy import ndimage
import multiprocessing as mp
from skimage import morphology
from skimage.measure import marching_cubes, mesh_surface_area

# import user defined library
from ShapeUtil.draw_lib import *
from ShapeUtil.data_structure import *
from Util.post_lib import check_folder_exist
from Util.parse_config import parse_config
from Util.segmentation_post_process import save_nii

warnings.filterwarnings("ignore")


def run_shape_analysis(process, config): # TODO: set GUI

    max_time = config["max_time"]

    mpPool = mp.Pool(mp.cpu_count()-1)
    configs = []
    for itime in tqdm(range(1, max_time+1), desc="Compose configs"):
        config['time_point'] = itime
        configs.append(config.copy())

    embryo_name = config["embryo_names"][0]
    for idx, _ in enumerate(tqdm(mpPool.imap_unordered(analyse_seg, configs), total=len(configs), desc="Naming {} segmentations".format(embryo_name))):
        process.emit('Collecting surface and volume', idx, max_time)

def analyse_seg(config):

    time_point = config['time_point']
    seg_file = os.path.join(config['seg_folder'], config['embryo_name'], config['embryo_name']+"_"+str(time_point).zfill(3)+'_segCell.nii.gz')
    seg = nib.load(seg_file).get_fdata()
    all_labels = np.unique(seg).tolist()
    all_labels.remove(0)

    nucleus_loc_to_save = pd.DataFrame.from_dict({"label": ["{}_{}".format(time_point, one_label) for one_label in all_labels],
                                                  "surface":[],
                                                  "volume":[]})
    seg = nib.load(seg_file).get_fdata()

    for one_label in all_labels:
        volume = (seg == one_label).sum() * (config["res"] ** 3)
        surface_area = get_surface_area(seg == one_label)
        nucleus_loc_to_save.loc[nucleus_loc_to_save.label == one_label, "surface_area"] = surface_area
        nucleus_loc_to_save.loc[nucleus_loc_to_save.label == one_label, "volume"] = volume

    # TODO: specify the the location here.
    save_name = os.path.join(config["project_folder"], 'SurfaceVolume', config['embryo_name'], str(config['time_point']).zfill(3) + '.csv')
    check_folder_exist(save_name)
    nucleus_loc_to_save.to_csv(save_name, index=False)
    # Save format:
    # Name          surface         volume
    # Time_label    ***             ****

    #  add connections between SegCell (edge and edge weight)
    contact_pairs, contact_ares = add_relation(seg) # TODO: save output as CSV
    # Save format
    # contact_pairs:    (label1, label2) ...;  contact_areas:   are

def add_relation(division_seg):
    '''
    Add relationship information between SegCell. (contact surface area)
    :param point_graph: point graph of SegCell
    :param division_seg: cell segmentations
    :return point_graph: contact graph between cells
    '''
    contact_pairs, contact_areas = get_contact_area(division_seg)

    return contact_pairs, contact_areas


def get_contact_area(volume):
    '''
    Get the contact volume surface of the segmentation. The segmentation results should be watershed segmentation results
    with a ***watershed line***.
    :param volume: segmentation result
    :return boundary_elements_uni: pairs of SegCell which contacts with each other
    :return contact_area: the contact surface area corresponding to that in the the boundary_elements.
    '''

    cell_mask = volume != 0
    boundary_mask = (cell_mask == 0) & ndimage.binary_dilation(cell_mask)
    [x_bound, y_bound, z_bound] = np.nonzero(boundary_mask)
    boundary_elements = []
    for (x, y, z) in zip(x_bound, y_bound, z_bound):
        neighbors = volume[np.ix_(range(x-1, x+2), range(y-1, y+2), range(z-1, z+2))]
        neighbor_labels = list(np.unique(neighbors))
        neighbor_labels.remove(0)
        if len(neighbor_labels) == 2:
            boundary_elements.append(neighbor_labels)
    boundary_elements_uni = list(np.unique(np.array(boundary_elements), axis=0))
    contact_area = []
    boundary_elements_uni_new = []
    for (label1, label2) in boundary_elements_uni:
        contact_mask = np.logical_and(ndimage.binary_dilation(volume == label1), ndimage.binary_dilation(volume == label2))
        contact_mask = np.logical_and(contact_mask, boundary_mask)
        if contact_mask.sum() > 4:
            verts, faces, _, _ = marching_cubes(contact_mask)
            area = mesh_surface_area(verts, faces) / 2
            contact_area.append(area)
            boundary_elements_uni_new.append((label1, label2))
    return boundary_elements_uni_new, contact_area

def get_surface_area(cell_mask):
    '''
    get cell surface area
    :param cell_mask: single cell mask
    :return surface_are: cell's surface are
    '''
    verts, faces, _, _ = marching_cubes(cell_mask)
    surface = mesh_surface_area(verts, faces)

    return surface