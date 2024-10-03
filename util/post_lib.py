# import dependency library
import os
import glob
import random
import numpy as np
import pickle
import pandas as pd
from treelib import Tree, Node

import nibabel as nib
import matplotlib.pyplot as plt
from itertools import combinations
from scipy import ndimage, stats
from scipy.spatial import Delaunay
from skimage.morphology import h_maxima
from scipy.ndimage import morphology, binary_dilation, binary_opening

def check_folder_exist(file_name):
    if not os.path.isdir(os.path.dirname(file_name)):
        os.makedirs(os.path.dirname(file_name))

def how_2D_array_as_image(image0):
    random.seed(1)
    image = (image0 / np.max(image0) * 250).astype(np.uint8)
    plt.imshow(image)
    plt.show()


def find_local_maximum(image, h):
    local_maximum_mask = h_maxima(image, h)

    return local_maximum_mask


def all_points_inline(x0, x1):
    d = np.diff(np.array((x0, x1)), axis=0)[0]
    j = np.argmax(np.abs(d))
    D = d[j]
    aD = np.abs(D)

    return x0 + (np.outer(np.arange(aD + 1), d) + (aD >> 1)) // (aD + 1e-6)


def line_weight_integral(x0, x1, weight_volume):
    # find all points between start and end
    inline_points = all_points_inline(x0, x1).astype(np.uint16)
    points_num = inline_points.shape[0]
    line_weight = 0
    for i in range(points_num):
        point_weight = weight_volume[inline_points[i][0],
                                    inline_points[i][1],
                                    inline_points[i][2]]
        line_weight = line_weight + point_weight

    return line_weight

def construct_weighted_graph(bin_image, local_max_h = 2):
    '''
    Construct edge weight graph from binary image.
    :param bin_image: cell binary image
    :return point_list: all points embedded in the triangulation, used for location query
    :return edge_list: list of edges in the triangulation
    :return edge_weight_list: edge weight corresponds to the edge list.
    '''
    volume_shape = bin_image.shape
    bin_cell = ndimage.morphology.binary_opening(bin_image).astype(np.float)
    bin_memb = bin_cell == 0
    bin_cell_edt = ndimage.morphology.distance_transform_edt(bin_cell)

    # get local maximum SegMemb
    local_maxima_mask = h_maxima(bin_cell_edt, local_max_h)
    [maxima_x, maxima_y, maxima_z] = np.nonzero(local_maxima_mask)
    #  find boundary points to force large weight
    x0 = np.where(maxima_x == 0)[0];
    x1 = np.where(maxima_x == volume_shape[0] - 1)[0]
    y0 = np.where(maxima_y == 0)[0];
    y1 = np.where(maxima_y == volume_shape[1] - 1)[0]
    z0 = np.where(maxima_z == 0)[0];
    z1 = np.where(maxima_z == volume_shape[2] - 1)[0]
    b_indx = np.concatenate((x0, y0, z0, x1, y1, z1), axis=None).tolist()
    point_list = np.stack((maxima_x, maxima_y, maxima_z), axis=1)
    tri_of_max = Delaunay(point_list)
    triangle_list = tri_of_max.simplices
    edge_list = []
    for i in range(triangle_list.shape[0]):
        for combine_pairs in combinations(triangle_list[i].tolist(), r=2):
            edge_list.append([combine_pairs[0], combine_pairs[1]])
    # add edges for all boundary points
    for i in range(len(b_indx)):
        for j in range(i, len(b_indx)):
            one_point = b_indx[i]
            another_point = b_indx[j]  #
            if ([one_point, another_point] in edge_list) or ([another_point, one_point] in edge_list):
                continue
            edge_list.append([one_point, another_point])

    weights_volume = bin_memb * 10000  # construct weights volume for graph
    edge_weight_list = []
    for one_edge in edge_list:
        start_x0 = point_list[one_edge[0]]
        end_x1 = point_list[one_edge[1]]
        if (one_edge[0] in b_indx) and (one_edge[1] in b_indx):
            edge_weight = 0  # All edges between boundary points are set as zero
        elif (one_edge[0] in b_indx) or (one_edge[1] in b_indx):
            edge_weight = 10000 * 10
        else:
            edge_weight = line_weight_integral(start_x0, end_x1, weights_volume)

        edge_weight_list.append(edge_weight)

    return point_list.tolist(), edge_list, edge_weight_list


def generate_graph_model(point_list, edge_list, edge_weight_list, img):
    '''
    Generate nii image for graph model
    :param point_list: local maximum list
    :param edge_list: edges list
    :param img: RawMemb image for shape
    :return:
    '''
    mask_max = np.zeros_like(img, np.uint8)
    tem_point_list = np.transpose(np.array(point_list), [1,0]).astype(np.uint8).tolist()
    mask_max[tem_point_list[0], tem_point_list[1], tem_point_list[2]] = 1
    mask_max = ndimage.morphology.binary_dilation(mask_max, iterations=5)
    valid_edge_volume = np.zeros_like(img, np.uint8)
    invalid_edge_volume = np.zeros_like(img, np.uint8)
    for i, one_edge in enumerate(edge_list):
        start_x0 = point_list[one_edge[0]]
        end_x1   = point_list[one_edge[1]]
        inline_points = all_points_inline(start_x0, end_x1)
        tem_point_list = np.transpose(np.array(inline_points), [1, 0]).astype(np.uint16).tolist()
        if edge_weight_list[i] < 10:
            valid_edge_volume[tem_point_list[0], tem_point_list[1], tem_point_list[2]] = 1
        else:
            invalid_edge_volume[tem_point_list[0], tem_point_list[1], tem_point_list[2]] = 1
    valid_edge_volume = ndimage.morphology.binary_dilation(valid_edge_volume)
    invalid_edge_volume = ndimage.morphology.binary_dilation(invalid_edge_volume)

    graph_model = np.zeros_like(img, np.uint8)
    graph_model[valid_edge_volume != 0] = 2  # for edges taht are preserved
    graph_model[invalid_edge_volume != 0] = 3 # for edges that are filterd
    graph_model[mask_max != 0] = 1  # for local maximum
    #nii_img = nib.Nifti1Image(np.transpose(graph_model, [2,1,0]), np.eye(4))

    return graph_model


def get_seconde_largest(img):
    '''
    Get the seconed largest connected component
    :param img:
    :return:
    '''
    label_post = ndimage.label(img)[0]
    count_label = np.bincount(label_post.flat)
    count_label[0] = 0  # delete the background number
    largestCC = (label_post == np.argmax(count_label)).astype(np.uint16)

    return largestCC


def set_boundary_zero(pre_seg):
    '''
    SET_BOUNARY_ZERO is used to set all segmented regions attached to the boundary as zero background.
    :param pre_seg:
    :return:
    '''
    opened_mask = binary_opening(pre_seg)
    pre_seg[opened_mask==0] = 0
    seg_shape = pre_seg.shape
    boundary_mask = np.zeros_like(pre_seg, dtype=np.uint8)
    boundary_mask[0:2, :, :] = 1; boundary_mask[:, 0:2, :] = 1; boundary_mask[:, :, 0:2] = 1
    boundary_mask[seg_shape[0]-1:, :, :] = 1; boundary_mask[:, seg_shape[1]-1:, :] = 1; boundary_mask[:, :, seg_shape[2]-1:] = 1
    boundary_labels = np.unique(pre_seg[boundary_mask != 0])
    for boundary_label in boundary_labels:
        pre_seg[pre_seg == boundary_label] = 0

    return pre_seg


def put_volume_constrain(segmentations, volume_limits):
    '''
    Put volume maximization to each region
    :param segmentations:
    :param volume_limits:
    :return:
    '''
    labels = np.unique(segmentations).tolist()
    labels.remove(0)

    # process lost nucleus in the background
    for key in volume_limits.keys():
        label = int(key)
        label_mask = segmentations == label
        label_edt = morphology.distance_transform_edt(label_mask)
        cell_mask = np.zeros_like(label_edt, dtype=bool)
        cell_mask[label_edt==np.amax(label_edt)] = True
        cell_mask = binary_dilation(cell_mask, iterations=4)
        nucleums_edt = morphology.distance_transform_edt(cell_mask==0)
        label_ma_edt = np.ma.masked_array(nucleums_edt, mask=label_mask==0)
        update_mask = find_inside_volume_mask(label_ma_edt, volume_limits[key], label_mask)
        if update_mask is None:
            continue
        segmentations[label_mask] = 0
        segmentations[update_mask] = label

    return segmentations


def find_inside_volume_mask(ma_mask, volume, label_mask):
    '''
    Volume SegMemb
    :param ma_mask:
    :param volume:
    :return:
    '''
    indexs = ma_mask.argsort(axis=None)[:int(volume - 10)]
    x, y, z = np.unravel_index(indexs, ma_mask.shape)
    update_mask = np.zeros(ma_mask.shape, dtype=bool)
    total_volume = label_mask.sum()
    if total_volume < volume:
        return None
    else:
        update_mask[x, y, z] = True

    return update_mask


def get_eggshell(wide_type_name):
    '''
    Get eggshell of specific embryo
    :param embryo_name:
    :return:
    '''
    wide_type_folder = os.path.join("./Data/MembTest", wide_type_name, "RawMemb")
    embryo_tp_list = glob.glob(os.path.join(wide_type_folder, "*.nii.gz"))
    overlap_num = 15 if len(embryo_tp_list) > 15 else len(embryo_tp_list)
    embryo_sum = nib.load(embryo_tp_list[0]).get_fdata()
    for tp_file in embryo_tp_list[1:overlap_num]:
        embryo_sum += nib.load(tp_file).get_fdata()

    embryo_mask = otsu3d(embryo_sum)
    valid_edt_mask0 = get_largest_connected_region(embryo_mask)
    dilated_mask = ndimage.morphology.binary_dilation(valid_edt_mask0, np.ones((3,3,3)), iterations=2)
    eggshell_mask = np.logical_and(dilated_mask, ~valid_edt_mask0)
    eggshell_mask[0:2, :, :] = False; eggshell_mask[:, 0:2, :] = False; eggshell_mask[:, :, 0:2] = False
    eggshell_mask[-3:, :, :] = False; eggshell_mask[:, -3:, :] = False; eggshell_mask[:, :, -3:] = False

    eggshell_mask = get_seconde_largest(eggshell_mask)

    return eggshell_mask.astype(np.uint8)


def otsu3d(gray):
    pixel_number = gray.shape[0] * gray.shape[1] * gray.shape[2]
    mean_weigth = 1.0/pixel_number
    his, bins = np.histogram(gray, np.arange(0,257))
    final_thresh = -1
    final_value = -1
    intensity_arr = np.arange(256)
    for t in bins[1:-1]: # This goes from 1 to 254 uint8 range (Pretty sure wont be those values)
        pcb = np.sum(his[:t])
        pcf = np.sum(his[t:])
        Wb = pcb * mean_weigth
        Wf = pcf * mean_weigth

        mub = np.sum(intensity_arr[:t]*his[:t]) / float(pcb)
        muf = np.sum(intensity_arr[t:]*his[t:]) / float(pcf)
        #print mub, muf
        value = Wb * Wf * (mub - muf) ** 2

        if value > final_value:
            final_thresh = t
            final_value = value
    final_img = gray.copy()
    final_img[gray > final_thresh] = 1
    final_img[gray < final_thresh] = 0

    return final_img


def get_largest_connected_region(embryo_mask):
    embryo_mask = ndimage.morphology.binary_opening(embryo_mask)
    label_structure = np.ones((3, 3, 3))
    [labelled_regions, _]= ndimage.label(embryo_mask, label_structure)
    [most_label, _] = stats.mode(labelled_regions, axis=None)
    valid_edt_mask0 = (labelled_regions == most_label[0])
    valid_edt_mask0 = ndimage.morphology.binary_closing(valid_edt_mask0)

    return valid_edt_mask0



def construct_celltree(nucleus_file, config):
    '''
    Construct cell tree structure with cell names
    :param nucleus_file:  the name list file to the tree initilization
    :param max_time: the maximum time point to be considered
    :return cell_tree: cell tree structure where each time corresponds to one cell (with specific name)
    '''

    ##  Construct cell
    #  Add unregulized naming
    cell_tree = Tree()
    cell_tree.create_node('P0', 'P0')
    cell_tree.create_node('AB', 'AB', parent='P0')
    cell_tree.create_node('P1', 'P1', parent='P0')
    cell_tree.create_node('EMS', 'EMS', parent='P1')
    cell_tree.create_node('P2', 'P2', parent='P1')
    cell_tree.create_node('P3', 'P3', parent='P2')
    cell_tree.create_node('C', 'C', parent='P2')
    cell_tree.create_node('P4', 'P4', parent='P3')
    cell_tree.create_node('D', 'D', parent='P3')
    cell_tree.create_node('Z2', 'Z2', parent='P4')
    cell_tree.create_node('Z3', 'Z3', parent='P4')
    cell_tree.create_node('ABa', 'ABa', parent='AB')
    cell_tree.create_node('ABp', 'ABp', parent='AB')
    cell_tree.create_node('ABal', 'ABal', parent='ABa')
    cell_tree.create_node('ABar', 'ABar', parent='ABa')
    cell_tree.create_node('ABpl', 'ABpl', parent='ABp')
    cell_tree.create_node('ABpr', 'ABpr', parent='ABp')


    # EMS
    cell_tree.create_node('E', 'E', parent='EMS')
    cell_tree.create_node('MS', 'MS', parent='EMS')

    # Read the name excel and construct the tree with complete SegCell
    df_time = pd.read_csv(nucleus_file)

    # read and combine all names from different acetrees
    ## Get cell number
    try:
        pd_number = pd.read_csv(config["number_dictionary"], names=["name", "label"])
        number_dictionary = pd.Series(pd_number.label.values, index=pd_number.name).to_dict()

    except:
        raise Exception("Name dictionary not found!")
        # ace_files = glob.glob('./ShapeUtil/AceForLabel/*.csv')
        # cell_list = [x for x in cell_tree.expand_tree()]
        # for ace_file in ace_files:
        #     ace_pd = pd.read_csv(os.path.join(ace_file))
        #     cell_list = list(ace_pd.cell.unique()) + cell_list
        #     cell_list = list(set(cell_list))
        # cell_list.sort()
        # number_dictionary = dict(zip(cell_list, range(1, len(cell_list)+1)))
        # with open(os.path.join(os.path.dirname(config["number_dictionary"]), 'number_dictionary.txt'), 'wb') as f:
        #     pickle.dump(number_dictionary, f)
        # with open(os.path.join(os.path.dirname(config["number_dictionary"]), 'name_dictionary.txt'), 'wb') as f:
        #     pickle.dump(dict(zip(range(1, len(cell_list)+1), cell_list)), f)

    max_time = len(os.listdir(os.path.join(config['seg_folder'], config['embryo_name'])))
    # max_time = config.get('max_time', 100)
    df_time = df_time[df_time.time <= max_time]
    all_cell_names = list(df_time.cell.unique())
    for cell_name in list(all_cell_names):
        if cell_name not in number_dictionary:
            continue
        times = list(df_time.time[df_time.cell==cell_name])
        cell_info = cell_node()
        cell_info.set_number(number_dictionary[cell_name])
        cell_info.set_time(times)
        if not cell_tree.contains(cell_name):
            if "Nuc" not in cell_name:
                parent_name = cell_name[:-1]
                cell_tree.create_node(cell_name, cell_name, parent=parent_name, data=cell_info)
        else:
            cell_tree.update_node(cell_name, data=cell_info)

    return cell_tree, max_time


class cell_node(object):
    # Node Data in cell tree
    def __init__(self):
        self.number = 0
        self.time = 0

    def set_number(self, number):
        self.number = number

    def get_number(self):

        return self.number

    def set_time(self, time):
        self.time = time

    def get_time(self):

        return self.time



def cell_prob_with_nucleus(cell, nucleus):
    '''
    This function is used to figure out whether one region is cell or empty hole (without nucleus)
    :param cell: segmentations results with different labels
    :param nucleus: nucleus RawMemb image (after resize)
    :return cell: cells without cavity
    :return hole: cavity inside the embryos
    '''

    labels = np.unique(cell).tolist()
    labels.remove(0)
    hole = np.zeros_like(cell, dtype=np.uint8)
    for label in labels:
        one_cell_mask = (cell == label)
        #  After checking on all intensity values, the segmented region should be regarded as empty when the intensity is
        #  lower than 100. Most are equals 0
        if (nucleus[one_cell_mask].sum() == 0):
            cell[one_cell_mask] = 0
            hole[one_cell_mask] = 1

    return cell, hole