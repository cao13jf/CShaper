

import os
import nibabel
import time
import numpy as np
import random
import codecs, json
from scipy import ndimage
from Util.post_lib import get_seconde_largest
from scipy import ndimage, stats
from skimage import morphology
import SimpleITK as sitk
import nibabel as nib

#===================================================================================#
#                           Basic file functions
#===================================================================================#
def search_file_in_folder_list(folder_list, file_name):
    """
    Find the full filename from a list of folders
    :param folder_list: a list of folders
    :param file_name:  filename
    :return full_file_name: the full filename
    """
    file_exist = False
    for folder in folder_list:
        full_file_name = os.path.join(folder, file_name)
        if(os.path.isfile(full_file_name)):
            file_exist = True
            break
    if(file_exist == False):
        raise ValueError('{0:} is not found in {1:}'.format(file_name, folder))
    return full_file_name

def load_3d_volume_as_array(filename):
    """
    load nifty file as array
    :param filename:file name
    :return array: 3D volume array
    """
    if('.nii' in filename):
        return load_nifty_volume_as_array(filename)
    raise ValueError('{0:} unspported file format'.format(filename))

def load_nifty_volume_as_array(filename, with_header = False):
    """
    load nifty image into numpy array, and transpose it based on the [z,y,x] axis order
    The output array shape is like [Depth, Height, Width]
    :param filename: the input file name, should be *.nii or *.nii.gz
    :param with_header: return affine and hearder infomation
    :return Data: a numpy Data array
    :return affine (optional): image affine
    :return head (optional): image header information
    """
    img = nibabel.load(filename)
    data = img.get_data()
    data = np.transpose(data, [2,1,0])
    if(with_header):
        return data, img.affine, img.header
    else:
        return data

def save_array_as_nifty_volume(data, filename, reference_name = None):
    """
    save a numpy array as nifty image
    :param Data: a numpy array with shape [Depth, Height, Width]
    :param filename: the ouput file name
    :param reference_name: file name of the reference image of which affine and header are used
    :return:
    """
    img = sitk.GetImageFromArray(data)
    folder_name = os.path.dirname(filename)
    if '.gz.gz' in filename:
        filename = filename[:-3]  # prevent possible input with '*.nii.gz'
    if not os.path.isdir(folder_name):
        os.makedirs(folder_name)
    if(reference_name is not None):
        img_ref = sitk.ReadImage(reference_name)
        img.CopyInformation(img_ref)
    sitk.WriteImage(img, filename)


def itensity_normalize_one_volume(volume):
    """
    normalize the itensity of an nd volume based on the mean and std of nonzeor region
    :param volume: the input nd volume
    :return out: the normalized nd volume
    """
    pixels = volume[volume > 0]
    mean = pixels.mean()
    std  = pixels.std()
    out = (volume - mean)/std
    out_random = np.random.normal(0, 1, size = volume.shape)
    out[volume == 0] = out_random[volume == 0]
    return out

def set_crop_to_volume(volume, bb_min, bb_max, sub_volume):
    """
    set a subregion to an nd image.
    :param volume: volume image
    :param bb_min: box region minimum
    :param bb_max: box region maximum
    :
    """
    dim = len(bb_min)
    out = volume
    if(dim == 2):
        out[np.ix_(range(bb_min[0], bb_max[0] + 1),
                   range(bb_min[1], bb_max[1] + 1))] = sub_volume
    elif(dim == 3):
        out[np.ix_(range(bb_min[0], bb_max[0] + 1),
                   range(bb_min[1], bb_max[1] + 1),
                   range(bb_min[2], bb_max[2] + 1))] = sub_volume
    elif(dim == 4):
        out[np.ix_(range(bb_min[0], bb_max[0] + 1),
                   range(bb_min[1], bb_max[1] + 1),
                   range(bb_min[2], bb_max[2] + 1),
                   range(bb_min[3], bb_max[3] + 1))] = sub_volume
    else:
        raise ValueError("array dimension should be 2, 3 or 4")
    return out

        
def get_random_crop_center(input_shape, output_shape):
    """
    get a random coordinate representing the center of a cropped volume
    :param input_shape: the shape of sampled volume
    :param output_shape: the desired crop shape
    :param center: the output center coordinate of a crop
    """
    center = []
    for i in range(len(input_shape)):
        x0 = int(output_shape[i]/2)
        x1 = input_shape[i] - x0
        if(x1 <= x0):
            centeri = int((x0 + x1)/2)
        else:
            centeri = random.randint(x0, x1)
        center.append(centeri)
    return center    

def transpose_volumes(volume, slice_direction):
    """
    transpose a list of volumes
    inputs:
        volumes: a list of nd volumes
        slice_direction: 'axial', 'sagittal', or 'coronal'
    outputs:
        tr_volumes: a list of transposed volumes
    """
    if (slice_direction == 'axial'):
        tr_volumes = volume
    elif(slice_direction == 'sagittal'):
        tr_volumes = np.transpose(volume, (2, 0, 1))
    elif(slice_direction == 'coronal'):
        tr_volumes = np.transpose(volume, (1, 0, 2))
    else:
        print('undefined slice direction:', slice_direction)
        tr_volumes = volume
    return tr_volumes


def transpose_volumes_reverse(volume, slice_direction):
    """
    transpose a list of volumes
    inputs:
        volumes: a list of nd volumes
        slice_direction: 'axial', 'sagittal', or 'coronal'
    outputs:
        tr_volumes: a list of transposed volumes
    """
    if (slice_direction == 'axial'):
        tr_volumes = volume
    elif(slice_direction == 'sagittal'):
        tr_volumes = np.transpose(volume, (1, 2, 0))
    elif(slice_direction == 'coronal'):
        tr_volumes = np.transpose(volume, (1, 0, 2))
    else:
        print('undefined slice direction:', slice_direction)
        tr_volumes = volume
    return tr_volumes

def crop_from_volume(volume, in_center, output_shape, fill = 'random'):
    """
    crop from a 3d volume
    :param volume: the input 3D volume
    :param in_center: the center of the crop
    :param output_shape: the size of the crop
    :param fill: 'random' or 'zero', the mode to fill crop region where is outside of the input volume
    :param output: the crop volume
    """
    input_shape = volume.shape   
    if(fill == 'random'):
        output = np.random.normal(0, 1, size = output_shape)
    else:
        output = np.zeros(output_shape)
    r0max = [int(x/2) for x in output_shape]  # If slicer center number is out of the range, it should be sliced based on shape
    r1max = [output_shape[i] - r0max[i] for i in range(len(r0max))]  # r0max=r1max when shape is even
    r0 = [min(r0max[i], in_center[i]) for i in range(len(r0max))]
    r1 = [min(r1max[i], input_shape[i] - in_center[i]) for i in range(len(r0max))]
    out_center = r0max

    # If there are valid layers in the volume, we sample with the center locating at the label_shape center. Otherwise,
    # layers outside of the volume are filled with random noise. In_center should always be the center at the new volume.
    output[np.ix_(range(out_center[0] - r0[0], out_center[0] + r1[0]),
                  range(out_center[1] - r0[1], out_center[1] + r1[1]),
                  range(out_center[2] - r0[2], out_center[2] + r1[2]))] = \
        volume[np.ix_(range(in_center[0] - r0[0], in_center[0] + r1[0]),
                      range(in_center[1] - r0[1], in_center[1] + r1[1]),
                      range(in_center[2] - r0[2], in_center[2] + r1[2]))]
    return output

def merge_crop_to_volume(volume, center, sub_volume):
    """
    merge the content of a crop to a 3d volume to a sub volume
    :param volume: the input 3D/4D volume
    :param center: the center of the crop
    :param sub_volume: the content of sub volume
    :param output_volume: the output 3D/4D volume
    """
    volume_shape = volume.shape   
    patch_shape = sub_volume.shape
    output_volume = volume
    for i in range(len(center)):
        if(center[i] >= volume_shape[i]):  # If the length on any dimension is bigger than the shape, return the whole
            return output_volume
    r0max = [int(x/2) for x in patch_shape]
    r1max = [patch_shape[i] - r0max[i] for i in range(len(r0max))]
    r0 = [min(r0max[i], center[i]) for i in range(len(r0max))]
    r1 = [min(r1max[i], volume_shape[i] - center[i]) for i in range(len(r0max))]
    patch_center = r0max

    if(len(center) == 3):
        output_volume[np.ix_(range(center[0] - r0[0], center[0] + r1[0]),
                             range(center[1] - r0[1], center[1] + r1[1]),
                             range(center[2] - r0[2], center[2] + r1[2]))] = \
            sub_volume[np.ix_(range(patch_center[0] - r0[0], patch_center[0] + r1[0]),
                              range(patch_center[1] - r0[1], patch_center[1] + r1[1]),
                              range(patch_center[2] - r0[2], patch_center[2] + r1[2]))]
    elif(len(center) == 4):
        output_volume[np.ix_(range(center[0] - r0[0], center[0] + r1[0]),
                             range(center[1] - r0[1], center[1] + r1[1]),
                             range(center[2] - r0[2], center[2] + r1[2]),
                             range(center[3] - r0[3], center[3] + r1[3]))] = \
            sub_volume[np.ix_(range(patch_center[0] - r0[0], patch_center[0] + r1[0]),
                              range(patch_center[1] - r0[1], patch_center[1] + r1[1]),
                              range(patch_center[2] - r0[2], patch_center[2] + r1[2]),
                              range(patch_center[3] - r0[3], patch_center[3] + r1[3]))]
    else:
        raise ValueError("array dimension should be 3 or 4")        
    return output_volume


def discrete_transform(continuous_img, non_linear = True, num_samples = 16):
    """
    Discretization on continous image.
    :param continuous_img: RawMemb image with continous value on pixel
    :param num_samples: the number of discretization
    :param discrete_img: discreted image
    """
    if non_linear:
        continuous_img  = np.exp(continuous_img)
    min_value = np.amin(continuous_img)
    max_value = np.amax(continuous_img)
    bins = np.linspace(min_value, max_value + np.finfo(float).eps, num=num_samples+1)
    discrete_img = np.digitize(continuous_img, bins) - 1


    return discrete_img


def binary_to_EDT_3D(binary_image, valid_edt_width, discrete_num_bins = 0):
    """
    Transform binary 3D SegMemb into distance transform SegMemb.
    :param binary_image: 3D bin
    :param EDT_image: distance transoformation of the image
    """
    assert len(binary_image.shape)==3, 'Input for EDT shoulb be 3D volume'
    if (discrete_num_bins==2):
        return binary_image
    edt_image = ndimage.distance_transform_edt(binary_image==0)

    # Cut out two large EDT far away from the binary SegMemb
    original_max_edt = np.max(edt_image)
    target_max_edt = min(original_max_edt, valid_edt_width)  # Change valid if given is too lagre
    valid_revised_edt = np.maximum(target_max_edt - edt_image, 0) / target_max_edt
    if(discrete_num_bins):
        discrete_revised_edt = discrete_transform(valid_revised_edt, non_linear=True, num_samples=discrete_num_bins)
    else:
        discrete_revised_edt = valid_revised_edt

    return discrete_revised_edt

def post_process_on_edt(edt_image):
    """
    Threshold the distance map and get the presegmentation results
    :param edt_image: distance map from EDT or net prediction
    :param final_seg: result after threshold
    """
    max_in_map = np.max(edt_image)
    assert max_in_map, 'Given threshold should be smaller the maximum in EDT'
    post_segmentation = np.zeros_like(edt_image, dtype=np.uint16)
    post_segmentation[edt_image == max_in_map] = 1
    largestCC = get_seconde_largest(post_segmentation)

    # Close operation on the thresholded image
    struct = ndimage.generate_binary_structure(3, 2)  # Generate binary structure for morphological operations
    final_seg = ndimage.morphology.binary_closing(largestCC, structure=struct).astype(np.uint16)

    return final_seg

def delete_isolate_labels(discrete_edt):
    '''
    delete all unconnected binary SegMemb
    '''
    label_structure = np.ones((3, 3, 3))
    [labelled_edt, _] = ndimage.label(discrete_edt == discrete_edt.max(), label_structure)

    # get the largest connected label
    [most_label, _] = stats.mode(labelled_edt[discrete_edt == discrete_edt.max()], axis=None)


    valid_edt_mask0 = (labelled_edt == most_label[0])
    valid_edt_mask = ndimage.morphology.binary_closing(valid_edt_mask0, iterations=2)
    filtered_edt = np.ones_like(discrete_edt)
    filtered_edt[valid_edt_mask == 0] = 0


    return filtered_edt


#===================================================================================#
#                               library for web GUI Data
#===================================================================================#
def save_numpy_as_json(np_data, save_file, surface_only = True):
    """
    Save python numpy Data as json for web GUI
    :param np_data: numpy variable (should be cell SegMemb embedded with embryo)
    :param save_file: save file name
    :param surface_only: whether exact the surface first and save surface points as json file
    :return:
    """
    if surface_only:
        np_data = get_cell_surface_mask(np_data)
    nonzero_loc = np.nonzero(np_data)
    nonzero_value = np_data[np_data!=0]
    loc_and_val = np.vstack(nonzero_loc + (nonzero_value,)).transpose().tolist()
    loc_and_val.insert(0, list((-1,) + np_data.shape))  # write volume size at the first location
    json.dump(loc_and_val, codecs.open(save_file, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)


def get_cell_surface_mask(cell_volume):
    """
    Extract cell surface SegMemb from the volume segmentation
    :param cell_volume: cell volume SegMemb with the membrane embedded
    :return cell_surface: cell surface with only surface pixels
    """
    cell_mask = cell_volume == 0
    strel = morphology.ball(2)
    dilated_cell_mask = ndimage.binary_dilation(cell_mask, strel, iterations=1)
    surface_mask = np.logical_and(~cell_mask, dilated_cell_mask)
    surface_seg = cell_volume
    surface_seg[~surface_mask] = 0

    return surface_seg

#===================================================================#
#                    For testing library function
#===================================================================#
if __name__=="__main__":
    start_time = time.time()
    seg = nib.load("/home/jeff/ProjectCode/LearningCell/DMapNet/ResultCell/test_embryo_robust/BinaryMembPostseg/181210plc1p2_volume_recovered/membT4CellwithMmeb.nii.gz").get_fdata()
    save_numpy_as_json(seg, "/home/jeff/ProjectCode/LearningCell/DMapNet/jsonSample4.json")
    print("runing time: {}s".format(time.time() - start_time))
