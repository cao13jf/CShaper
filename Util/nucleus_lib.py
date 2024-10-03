'''
This library is used to incorporate
'''

import numpy as np


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
