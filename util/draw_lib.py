'''
Library for drawing results during shape analysis stage
'''
import os
import cv2
import colorsys
import random
import numpy as np
import nibabel as nib
import networkx as nx
import matplotlib.pyplot as plt


def draw_relation_graph(relation_graph, nuc_position):

    # nx.draw(relation_graph, pos=nuc_position, with_labels=True, node_size=100, font_color='b',
    edges, weights = zip(*nx.get_edge_attributes(relation_graph, 'area').items())

    nx.draw(relation_graph, node_size=100, edgelist=edges, edge_color=weights, width=3, \
            edge_cmap=plt.cm.Blues, with_labels=True)
    # plt.savefig('edges.png')


def make_video(images, cell_nums, times, outvid='./slice_video.avi', fps=5, size=None,
               is_color=True, format="XVID"):
    """
    Create a video from a list of images.

    @param      images      list of images to use in the video
    @param      outvid      output video
    @param      fps         frame per second
    @param      size        size of each frame
    @param      is_color    color
    @param      format      see http://www.fourcc.org/codecs.php
    @return                 see http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html

    The function relies on http://opencv-python-tutroals.readthedocs.org/en/latest/.
    By default, the video will have the size of the first image.
    It will resize every image to this size before adding them to the video.
    """

    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    font = cv2.FONT_HERSHEY_SIMPLEX
    vid = None
    for image, cell_num, time in zip(images, cell_nums, times):
        img = image
        if vid is None:
            if size is None:
                size = img.shape[1], img.shape[0]
                #writer = cv2.VideoWriter('./output.avi', cv2.VideoWriter_fourcc('P','I','M','1'), 25, (100,100), True)
            vid = cv2.VideoWriter(outvid, fourcc, float(fps), size, is_color)
        if size[0] != img.shape[1] and size[1] != img.shape[0]:
            img = cv2.resize(img, size)
        if is_color:
            cv2.putText(img, '#Cell=' + str(cell_num)+'  #T='+str(time), (20, 20), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        else:
            cv2.putText(img, '#Cell=' + str(cell_num)+'  #T='+str(time), (20, 20), font, 0.5, 255, 1, cv2.LINE_AA)
        vid.write(img)
    vid.release()
    a = 2


def generate_video(file_dir, raw_dir, start_time=1, end_time=90, slice_num=60):
    '''
    Generate slice video from start time to end_time
    :param start_time:
    :param end_time:
    :param slice_num:
    :return:
    '''
    images = []
    raw_images = []
    im_combs = []
    cell_nums = []
    lut = generate_lut()
    for time_point in range(start_time, end_time):
        file_name = os.path.join(file_dir, f'membT{time_point}CellwithMmeb.nii.gz')
        img = nib.load(file_name)
        image_data = img.get_fdata().astype(np.uint16)
        cell_nums.append(np.unique(image_data).shape[0]-1)
        image_data_256 = (image_data % 255).astype(np.uint8)
        image_data_256[np.logical_and(image_data_256 == 0, image_data != 0)] = 2
        im_color = cv2.LUT(np.repeat(image_data_256[:, :, slice_num, np.newaxis], 3, axis=-1), lut)
        #im_color = cv2.applyColorMap(image_data_256[:, :, slice_num], cv2.COLORMAP_HSV)
        images.append(im_color)

        raw_file_name = os.path.join(raw_dir, f'membT{time_point}.nii.gz')
        img = nib.load(raw_file_name)
        image_data = img.get_fdata().astype(np.uint8)
        img_raw = image_data[:,:,slice_num]
        raw_images.append(img_raw)

        # Combine the RawMemb and gray image into one frame
        rows_rgb, cols_rgb, channels = im_color.shape
        im_comb = np.zeros(shape=(rows_rgb, cols_rgb*2, channels), dtype=np.uint8)
        im_comb[:, cols_rgb:] = im_color
        im_comb[:, :cols_rgb] = img_raw[:, :, None]
        im_combs.append(im_comb)

    make_video(im_combs, cell_nums, list(range(start_time, end_time)), './slice_video_170704plc1p2.avi', is_color=True)
    # make_video(raw_images, cell_nums, './slice_video_raw.avi', is_color=False)

def generate_lut(N=256):
    '''
    Generate a look up table with N different colors.
    :param N:
    :return:
    '''
    HSV_tuples = [[x * 1.0 / N, 1, 1] for x in range(N)]
    random.shuffle(HSV_tuples)
    RGB_tuples = list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))
    lut = np.expand_dims(np.array(RGB_tuples), axis=0)
    lut[0, 0, :] = 0

    return (lut * 255).astype(np.uint8)


if __name__== '__main__':

    # For segmentation
    generate_video('../ResultCell/edt_discrete/BinaryMembPostseg/170704plc1p2_label_unified',
                   '../Data/MembTest/170704plc1p2/RawMemb', start_time=1, end_time=120, slice_num=60)