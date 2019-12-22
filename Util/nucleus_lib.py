'''
This library is used to incorporate
'''

import numpy as np


def cell_prob_with_nucleus(cell, nucleus):
    '''
    This function is used to figure out whether one region is cell or empty hole (without nucleus)
    :param cell: segmentations results with different labels
    :param nucleus: nucleus rawMemb image (after resize)
    :return cell_prob:  probability of whether segmented region being a valid cell
    '''

    nucleus = nucleus.astype(np.float32)
    nucleus = nucleus / nucleus.max()
    labels = np.unique(cell).tolist()
    labels.remove(0)

    label_intensity = []
    for label in labels:
        one_cell_mask = (cell == label)
        nucleus_intensity = nucleus[one_cell_mask]
        #  After checking on all intensity values, the segmented region should be regarded as empty when the intensity is
        #  lower than 100. Most are equals 0
        label_intensity.append(nucleus_intensity.sum())
        if (nucleus_intensity.sum() < 10):
            cell[one_cell_mask] = 0

    return cell
