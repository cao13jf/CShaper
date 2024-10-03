'''data augmentation'''
# import dependency library
import random
import math
from scipy.ndimage.morphology import binary_dilation
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt

#=============================================
#   augmentation for raw data
#=============================================


#=============================================
#   augmentation for label
#=============================================
#  get sliced distance transform which cross the center of nucleus.
def contour_distance(contour_label, d_threshold=15):
    '''
    :param label:  Binary label of the target
    :param center_stack:
    :return:
    '''
    background_edt = distance_transform_edt(contour_label == 0)
    background_edt[background_edt > d_threshold] = d_threshold
    vertical_slice_edt = (d_threshold - background_edt) / d_threshold

    return vertical_slice_edt.astype(np.float32)

def contour_distance_outside_negative(contour_label, inside_mask, d_threshold=15):
    '''Distance transform with positive d inside and negative outside
    :param contour_label: contour boundary of each region
    :param inside_mask: inside regions mask (without contour mask)
    :param d_threshold: distance threshold which should be cut off
    :return distance: distance map with positive inside, negative outside and zero height membrane
    '''

    #background normalized distance
    inside_region_and_contour = np.logical_or(contour_label, inside_mask)
    background_edt = distance_transform_edt(~inside_region_and_contour)
    background_edt[background_edt > d_threshold] = d_threshold
    background_edt_normalized = background_edt / d_threshold

    #inside normalized distance based on membrane contour
    inside_edt = distance_transform_edt(contour_label == 0)
    inside_edt[inside_edt > d_threshold] = d_threshold
    inside_edt_normalized = inside_edt / d_threshold

    #combine inside and outside distance with positive and negative difference
    inside_edt_normalized[~inside_region_and_contour] = 0 - background_edt_normalized[~inside_region_and_contour]
    return inside_edt_normalized.astype(np.float32)

#  cell centered distance transform
def cell_sliced_distance(seg_cell, seg_nuc, sampled=True, d_threshold=15):
    # sampled cell labels
    cell_labels = np.unique(seg_cell)[1:].tolist()
    sampled_num = 10 if len(cell_labels) > 10 else len(cell_labels)
    sampled_labels = random.sample(cell_labels, k=sampled_num)
    cell_mask = np.isin(seg_cell, sampled_labels)
    #  edt transformation
    vertical_slice_edt = distance_transform_edt(seg_nuc == 0)
    vertical_slice_edt[vertical_slice_edt > d_threshold] = d_threshold
    vertical_slice_edt = (d_threshold - vertical_slice_edt) / d_threshold
    # vertical_slice_edt = add_volume_boundary_mask(vertical_slice_edt, fill_value=0)
    # #  to simulate slice annotation, only keep slices through the nucleus #
    # keep_mask = np.zeros_like(cell_mask, dtype=bool)
    # for label in sampled_labels:
    #     tem_mask = np.zeros_like(cell_mask, dtype=bool)
    #     single_cell_mask = (seg_cell==label)
    #     x, y, z = np.nonzero(np.logical_and(seg_nuc, single_cell_mask))
    #     tem_mask[x, :, :] = True; tem_mask[:, y, :] = True; tem_mask[:, :, z] = True  # still too many slices annotation
    #     # combine different cells
    #     keep_mask = np.logical_or(keep_mask, np.logical_and(single_cell_mask, tem_mask))
    #
    # vertical_slice_edt[~keep_mask] = 0  # Output -1 for less attetion in loss
    keep_mask = np.ones_like(vertical_slice_edt)
    return vertical_slice_edt.astype(np.float), (keep_mask).astype(np.float)   # output keep_mask to count on valid mask

#  sample partials binary cell boundary and cell mask for weak supervision
def sampled_cell_mask(seg_cell, seg_memb):
    cell_labels = np.unique(seg_cell)[1:].tolist()
    sampled_num = 200 if len(cell_labels) > 200 else len(cell_labels)
    sampled_labels = random.sample(cell_labels, k=sampled_num)
    cell_mask = np.isin(seg_cell, sampled_labels)
    strl = np.ones((3, 3, 3), dtype=bool)
    keep_mask = binary_dilation(cell_mask, structure=strl)
    cell_mask = np.logical_and(keep_mask, seg_memb)

    return cell_mask.astype(np.float), keep_mask.astype(np.float)

#  change regression data to discrete class data
def regression_to_class(res_data, out_class, uniform=True):
    bins = np.arange(out_class) / (out_class - 1)
    if not uniform:
        bins = np.sqrt(bins)

    return np.digitize(res_data, bins, right=True)

#  add prior information to boundary mask
def add_volume_boundary_mask(data, fill_value):
    W, H, D = data.shape
    data[[0, W-1], :, :] = fill_value
    data[:, [0, H-1], :] = fill_value
    data[:, :, [0, D-1]] = fill_value

    return data

