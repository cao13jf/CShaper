
#  import dependent library
import os
import glob
import random
# import open3d as o3d
from tqdm import tqdm
import numpy as np
import pandas as pd
import multiprocessing as mp
from scipy import ndimage, stats
import nibabel as nib
import pickle
from skimage.morphology import h_maxima, binary_opening
from skimage.segmentation import watershed
from scipy.ndimage.morphology import binary_closing, distance_transform_edt
from skimage.measure import marching_cubes, mesh_surface_area
from scipy.spatial import Delaunay
from scipy.stats import mode

# import user defined library
from Util.data_process import itensity_normalize_one_volume, load_3d_volume_as_array, save_array_as_nifty_volume
from Util.data_utils import check_folder
from Util.segmentation_post_process import save_nii
from ShapeUtil.data_structure import construct_celltree

#=========================================================================
#   main function for post process
#=========================================================================
def nib_load(file_name):
    if not os.path.exists(file_name):
        raise IOError("Cannot find file {}".format(file_name))
    return nib.load(file_name).get_fdata()

def nib_save(data, file_name):
    check_folder(file_name)
    return nib.save(nib.Nifti1Image(data, np.eye(4)), file_name)

# def generate_alpha_shape(points_np: np.array, displaying: bool = False, alpha_value: float = 0.88, view_name: str = 'default', print_info: bool = False):
#     '''
#
#     :param points_np:
#     :param displaying:
#     :param alpha_value: bigger than 0.85, sqrt(3)/2, the delaunay triangulation would be successful.
#         Because of the random bias, we need to add more than 0.01 to the 0.866
#     :param view_name:
#     :return:
#     '''
#     m_pcd = o3d.geometry.PointCloud()
#
#     m_pcd.points = o3d.utility.Vector3dVector(points_np)  # observe the points with np.asarray(pcd.points)
#
#     # the mesh is developed from http://www.open3d.org/docs/release/python_api/open3d.geometry.TriangleMesh.html
#     m_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(m_pcd, alpha_value)
#
#     if print_info:
#         print('mesh info', m_mesh)
#         print('edge manifold', m_mesh.is_edge_manifold(allow_boundary_edges=True))
#         print('edge manifold boundary', m_mesh.is_edge_manifold(allow_boundary_edges=False))
#         print('vertex manifold', m_mesh.is_vertex_manifold())
#         print('watertight', m_mesh.is_watertight())
#         print(f"alpha={alpha_value:.3f}")
#
#     if displaying:
#         # add normals, add light effect
#         m_mesh.compute_vertex_normals()
#
#         # make the non manifold vertices become red
#         vertex_colors = 0.75 * np.ones((len(m_mesh.vertices), 3))
#         for boundary in m_mesh.get_non_manifold_edges():
#             for vertex_id in boundary:
#                 vertex_colors[vertex_id] = [1, 0, 0]
#         m_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
#
#         o3d.visualization.draw_geometries([m_mesh], mesh_show_back_face=True, mesh_show_wireframe=True,
#                                           window_name=view_name)
#     return m_mesh

def segment_membrane(para):
    '''

    :param para:
    :return:
    '''
    # transform the binary segmentation result to the single cell
    embryo_name = para[0]
    this_niigz_file_name = para[1] # the path of segMemb file (binary segmentation result)
    egg_shell = para[2] # the binary mask of a embryo with histogram transform ;todo: not used here?
    running_data_dir=para[3]

    embryo_name_tp = "_".join(os.path.basename(this_niigz_file_name).split("_")[0:2])
    # the segNuc file have only one pixel for each nuc, Dr. Cao will dialation to bigger (radius 8)
    segNuc_file = os.path.join(running_data_dir.replace("CellMembrane", "RawStack"), embryo_name, "SegNuc", embryo_name_tp + "_segNuc.nii.gz")

    memb_edt = load_3d_volume_as_array(this_niigz_file_name)# * egg_shell
    # =============== change to binary front distance ========================
    # center_mask = memb_edt > (memb_edt.max() * 0.90)
    # center_mask = get_largest_connected_region(center_mask)
    # center_distance = distance_transform_edt(~center_mask)
    # memb_edt = center_distance.max() - center_distance

    marker_volume = load_3d_volume_as_array(segNuc_file)

    # marker_volume[memb_edt > 250] = 0  # delete died cells
    marker_volume = ndimage.morphology.grey_dilation(marker_volume, structure=np.ones((7, 7, 7))) - 1
    # Combine sisters

    # add background seeds
    memb_edt[0:3, :, :] = 0; memb_edt[:, 0:3, :] = 0; memb_edt[:, :, 0:3] = 0
    memb_edt[-4:-1, :, :] = 0; memb_edt[:, -4:-1, :] = 0; memb_edt[:, :, -4:-1] = 0
    marker_volume[0:2, :, :] = 10000; marker_volume[:, 0:2, :] = 10000; marker_volume[:, :, 0:2] = 10000
    marker_volume[-3:-1, :, ] = 10000; marker_volume[:, -3:-1, :] = 10000; marker_volume[:, :, -3:-1] = 10000

    watershed_seg = watershed(memb_edt, marker_volume.astype(np.uint16), watershed_line=True)
    #  set background label as zero
    watershed_seg[watershed_seg == 10000] = 0
    # merged_seg = set_boundary_zero(merged_seg)

    save_name = os.path.join(running_data_dir + "Postseg", embryo_name, embryo_name_tp+"_segCell.nii.gz")
    save_array_as_nifty_volume(watershed_seg.astype(np.uint16), save_name)


#================================================================================
#       extra tools
#================================================================================
#  construct local maximum graph
def construct_weighted_graph(bin_image, local_max_h = 2):
    '''
    Construct edge weight graph from binary MembAndNuc.
    :param bin_image: cell binary MembAndNuc
    :return point_list: all points embedded in the triangulation, used for location query
    :return edge_list: list of edges in the triangulation
    :return edge_weight_list: edge weight corresponds to the edge list.
    '''

    volume_shape = bin_image.shape
    bin_cell = ndimage.morphology.binary_opening(bin_image).astype(float)
    bin_memb = bin_cell == 0
    bin_cell_edt = ndimage.morphology.distance_transform_edt(bin_cell)

    # get local maximum mask
    local_maxima_mask = h_maxima(bin_cell_edt, local_max_h)
    [maxima_x, maxima_y, maxima_z] = np.nonzero(local_maxima_mask)
    #  find boundary points to force large weight
    x0 = np.where(maxima_x == 0)[0]; x1 = np.where(maxima_x == volume_shape[0] - 1)[0]
    y0 = np.where(maxima_y == 0)[0]; y1 = np.where(maxima_y == volume_shape[1] - 1)[0]
    z0 = np.where(maxima_z == 0)[0]; z1 = np.where(maxima_z == volume_shape[2] - 1)[0]
    b_indx = np.concatenate((x0, y0, z0, x1, y1, z1), axis=None).tolist()


    point_list = np.stack((maxima_x, maxima_y, maxima_z), axis=1)
    tri_of_max = Delaunay(point_list)
    triangle_list = tri_of_max.simplices
    edge_list = []
    for i in range(triangle_list.shape[0]):
        for j in range(triangle_list.shape[1]-1):
            edge_list.append([triangle_list[i][j-1], triangle_list[i][j]])
        edge_list.append([triangle_list[i][j], triangle_list[i][0]])
    # add edges for all boundary points
    for i in range(len(b_indx)):
        for j in range(i, len(b_indx)):
            one_point = b_indx[i]
            another_point = b_indx[j]
            if ([one_point, another_point] in edge_list) or ([another_point, one_point] in edge_list):
                continue
            edge_list.append([one_point, another_point])

    weights_volume = bin_memb * 10000  # construct weights volume for graph
    edge_weight_list = []
    for one_edge in edge_list:
        start_x0 = point_list[one_edge[0]]
        end_x1   = point_list[one_edge[1]]
        if (one_edge[0] in b_indx) and (one_edge[1] in b_indx):
            edge_weight = 0  # All edges between boundary points are set as zero
        elif (one_edge[0] in b_indx) or (one_edge[1] in b_indx):
            edge_weight = 10000 * 10
        else:
            edge_weight = line_weight_integral(start_x0, end_x1, weights_volume)

        edge_weight_list.append(edge_weight)

    return point_list.tolist(), edge_list, edge_weight_list



    # backup points that should be labelled together
def combine_background_maximum(point_weight_list):
    point_tomerge_list0 = []
    for one_edge in point_weight_list:
        added_flag = 0
        point1 = one_edge[0]
        point2 = one_edge[1]
        for i in range(len(point_tomerge_list0)):
            if (point1 in point_tomerge_list0[i]) or (point2 in point_tomerge_list0[i]):
                point_tomerge_list0[i] = list(set().union([point1, point2], point_tomerge_list0[i]))
                added_flag = 1
                break
        if not added_flag:
            point_tomerge_list0.append([point1, point2])

    return point_tomerge_list0

    #  combine local maximums inside the embryo
def combine_inside_maximum(cluster1, cluster2):
    point_tomerge_list = []
    merged_cluster = []
    while len(cluster1):
        delete_index = []
        cluster_in1 = cluster1.pop()
        if cluster_in1 in merged_cluster:
            continue
        cluster_final = set(cluster_in1)
        for cluster_in2 in cluster2:
            tem_final = set(cluster_final).intersection(cluster_in2)
            if len(tem_final):
                merged_cluster.append(cluster_in2)
                cluster_final = set().union(cluster_final, cluster_in2)
        point_tomerge_list.append(list(cluster_final))
    return point_tomerge_list

#  reverse over-segmentation with merged maximum.
def reverse_seg_with_max_cluster(watershed_seg, init_list, max_cluster):
    merged_seg = watershed_seg.copy()
    for one_merged_points in max_cluster:
        first_point = init_list[one_merged_points[0]]
        one_label = watershed_seg[first_point[0], first_point[1], first_point[2]]
        for other_point in one_merged_points[1:]:
            point_location = init_list[other_point]
            new_label = watershed_seg[point_location[0], point_location[1], point_location[2]]
            merged_seg[watershed_seg == new_label] = one_label
        one_mask = merged_seg == one_label
        one_mask_closed = ndimage.binary_closing(one_mask)
        merged_seg[one_mask_closed!=0] = one_label

    return merged_seg

#  set all boundary torched labels as zero
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

#   filter over-segmentation with binary nucleus loc
def cell_filter_with_nucleus(cell_seg, nuc_seg):
    init_all_labels = np.unique(cell_seg).tolist()
    keep_labels = (np.unique(cell_seg[nuc_seg > 0])).tolist()
    filtered_labels = list(set(init_all_labels) - set(keep_labels))
    for label in filtered_labels:
        cell_seg[cell_seg == label] = 0

    return cell_seg

def get_largest_connected_region(embryo_mask):
    '''
    erase the little noise
    :param embryo_mask:
    :return:
    '''
    label_structure = np.ones((3, 3, 3))
    embryo_mask = ndimage.morphology.binary_closing(embryo_mask, structure=label_structure)
    [labelled_regions, _]= ndimage.label(embryo_mask, label_structure)
    count_label = np.bincount(labelled_regions.flat)
    count_label[0] = 0
    valid_edt_mask0 = (labelled_regions == np.argmax(count_label))
    valid_edt_mask0 = ndimage.morphology.binary_closing(valid_edt_mask0)

    return valid_edt_mask0.astype(np.uint8)

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
        neighbors = volume[np.ix_(range(x-1, x+2),
                                         range(y-1, y+2),
                                         range(z-1, z+2))]
        neighbor_labels = list(np.unique(neighbors))
        neighbor_labels.remove(0)
        if len(neighbor_labels)==2:
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
    # ball_structure = morphology.cube(3) #
    # erased_mask = ndimage.binary_erosion(cell_mask, ball_structure, iterations=1)
    # surface_area = np.logical_and(~erased_mask, cell_mask).sum()
    verts, faces, _, _ = marching_cubes(cell_mask)
    surface = mesh_surface_area(verts, faces)

    return surface

def delete_isolate_labels(discrete_edt):
    '''
    delete all unconnected binary SegMemb
    '''
    label_structure = np.ones((3, 3, 3))
    [labelled_edt, _]= ndimage.label(discrete_edt, label_structure)

    # get the largest connected label
    [most_label, _] = stats.mode(labelled_edt[discrete_edt == discrete_edt.max()], axis=None)

    valid_edt_mask0 = (labelled_edt == most_label[0])
    valid_edt_mask = ndimage.morphology.binary_closing(valid_edt_mask0, iterations=2)
    filtered_edt = np.copy(discrete_edt)
    filtered_edt[valid_edt_mask == 0] = 0


    return filtered_edt
#================================================================
#   get egg shell
#================================================================
def get_eggshell(wide_type_name, root_folder=r'./dataset/test',hollow=False):
    '''
    Get eggshell of specific embryo
    :param embryo_name:
    :return:
    '''
    wide_type_folder = os.path.join(root_folder, wide_type_name, "RawMemb")
    embryo_tp_list = glob.glob(os.path.join(wide_type_folder, "*.nii.gz"))
    random.shuffle(embryo_tp_list)
    overlap_num = 15 if len(embryo_tp_list) > 15 else len(embryo_tp_list)
    embryo_sum = nib_load(embryo_tp_list[0]).astype(np.float)
    for tp_file in embryo_tp_list[1:overlap_num]:
        embryo_sum += nib_load(tp_file) # add the random 15 images, why?????

    embryo_mask = otsu3d(embryo_sum) # building a binary mask with histogram transform with sum of 15 raw images.
    embryo_mask = get_largest_connected_region(embryo_mask)
    embryo_mask[0:2, :, :] = False; embryo_mask[:, 0:2, :] = False; embryo_mask[:, :, 0:2] = False
    embryo_mask[-3:, :, :] = False; embryo_mask[:, -3:, :] = False; embryo_mask[:, :, -3:] = False
    if hollow:
        dilated_mask = ndimage.morphology.binary_dilation(embryo_mask, np.ones((3,3,3)), iterations=2)
        eggshell_mask = np.logical_and(dilated_mask, ~embryo_mask)
        return eggshell_mask.astype(np.uint8)
    return embryo_mask.astype(np.uint8)


def otsu3d(gray):
    # added 15 raw membrane gii.gz
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


'''
Based on the division summary, all cells should be seperated at 4th TP
'''
# def combine_division(embryos, max_times, running_dir,overwrite=False):
#     # judge if embryos a list
#     embryos = embryos if isinstance(embryos, list) else [embryos]
#
#     # # todo: Danger usage 1 - here. But time is limited, use this first
#     # with open(os.path.join('tem_files','wrong_division_cells.pikcle'), "rb") as fp:  # Unpickling
#     #     list_wrong_division = pickle.load(fp)
#
#     for i_embryo, embryo in enumerate(embryos):
#         max_time = max_times[i_embryo]
#         ace_file = os.path.join(running_dir, 'CDFiles', "CD"+embryo+".csv") # pixel x y z
#         cell_tree, max_time = construct_celltree(ace_file, max_time=max_time)
#
#         try:
#             pd_number = pd.read_csv('./dataset/number_dictionary.csv', names=["name", "label"])
#             number_dict = pd.Series(pd_number.label.values, index=pd_number.name).to_dict()
#         except:
#             raise Exception("Not find number dictionary at ./dataset")
#         name_dict = dict((v, k) for k, v in number_dict.items())
#
#         memb_files = glob.glob(os.path.join(running_dir, embryo, "SegMemb", '*.nii.gz'))
#         cell_files = glob.glob(os.path.join(running_dir, embryo, "SegCell", '*.nii.gz'))
#         nuc_files = glob.glob(os.path.join(running_dir, embryo, "SegNuc", '*.nii.gz')) # acetree editting result
#         memb_files.sort()
#         nuc_files.sort()
#         cell_files.sort()
#
#         # ================== single test =================================
#         for tp_indx in tqdm(range(0, len(nuc_files), 1), desc="Combining the dividing cell in SegCellDividingCells) {}".format(embryo)):
#             embryo = embryo # todo ? : what is this
#             memb_file = memb_files[tp_indx] # prediction result, binary segmentation result
#             nuc_file = nuc_files[tp_indx]
#             cell_file = cell_files[tp_indx]
#
#             exact_this_frame_tp=int(os.path.basename(memb_file).split('_')[1])
#
#             seg_map = nib_load(memb_file)
#             seg_memb_edt = (seg_map > 0.93 * seg_map.max()).astype(float)
#             labbel_seg_nuc = nib_load(nuc_file)
#             seg_cell = nib_load(cell_file)
#
#             labels = np.unique(labbel_seg_nuc).tolist()
#             labels.pop(0)
#             processed_labels = []
#             output_seg_cell = seg_cell.copy()
#             cell_labels = np.unique(seg_cell).tolist()
#             # start to detect dividing cells
#             for one_label in labels:
#                 one_times = cell_tree[name_dict[one_label]].data.get_time()
#                 # only deal with the later time point, the former is dealt at the former loop
#                 if any(time < exact_this_frame_tp for time in one_times):
#                     continue
#                 # no need to deal with the sister cells
#                 if (one_label in processed_labels):
#                     continue
#                 parent_label = cell_tree.parent(name_dict[one_label])
#                 if parent_label is None:
#                     continue
#                 # sister cell
#                 another_label = [number_dict[a.tag] for a in cell_tree.children(parent_label.tag)]
#                 another_label.remove(one_label)
#                 another_label = another_label[0]
#                 # this one cell label and sister cell label should both in this time point this volume
#                 if (one_label not in cell_labels) or (another_label not in cell_labels):
#                     continue
#
#                 this_cell_nucleus_pos = np.stack(np.where(labbel_seg_nuc == one_label)).squeeze().tolist()
#                 sister_cell_nucleus_pos = np.stack(np.where(labbel_seg_nuc == another_label)).squeeze().tolist()
#                 edge_weight = line_weight_integral(pos1=this_cell_nucleus_pos, pos2=sister_cell_nucleus_pos, weight_volume=seg_memb_edt)
#                 if edge_weight == 0:
#                     mask = np.logical_or(seg_cell == one_label, seg_cell == another_label)
#                     # erase the gap between two cells
#                     mask = binary_closing(mask, structure=np.ones((3, 3, 3)))
#
#                     if exact_this_frame_tp>100:
#                         # ---------------validate this these two cells belong to two regions------------------------
#                         tuple_tmp = np.where(mask)
#                         # print(len(tuple_tmp))
#                         sphere_list = np.concatenate(
#                             (tuple_tmp[0][:, None], tuple_tmp[1][:, None], tuple_tmp[2][:, None]), axis=1)
#                         adjusted_rate = 0.01
#                         sphere_list_adjusted = sphere_list.astype(float) + np.random.uniform(0, adjusted_rate,
#                                                                                              (len(tuple_tmp[0]), 3))
#                         alpha_v = 1.5
#                         m_mesh = generate_alpha_shape(sphere_list_adjusted, alpha_value=alpha_v)
#                         # print(len(m_mesh.cluster_connected_triangles()[2]))
#
#                         # alpha_v = 1
#
#                         if len(m_mesh.cluster_connected_triangles()[2]) > 1:
#                             continue
#
#
#                     output_seg_cell[mask] = number_dict[parent_label.tag]
#                     one_times.remove(exact_this_frame_tp)
#                     another_times = cell_tree[name_dict[another_label]].data.get_time()
#                     another_times.remove(exact_this_frame_tp)
#                     cell_tree[name_dict[one_label]].data.set_time(one_times)
#                     cell_tree[name_dict[another_label]].data.set_time(another_times)
#                 processed_labels += [one_label, another_label]
#
#             if not overwrite:
#                 seg_save_file = os.path.join(running_dir, embryo, "SegCellDividingCells",
#                                              embryo + "_" + str(exact_this_frame_tp).zfill(3) + "_segCell.nii.gz")
#             else:
#                 seg_save_file = cell_file
#             nib_save(output_seg_cell, seg_save_file)
        # =============== single test =================================

        # parameters = []
        # for i, file_name in enumerate(memb_files):
        #     parameters.append([embryo, file_name, nuc_files[i], cell_files[i], cell_tree, overwrite, number_dict, name_dict])
        #
        #     # combine_division_mp([embryo, file_name, nuc_files[i], cell_files[i], cell_tree, overwrite, number_dict, name_dict])
        # mpPool = mp.Pool(mp.cpu_count() - 1)
        # for _ in tqdm(mpPool.imap_unordered(combine_division_mp, parameters), total=len(parameters), desc=embryo):
        #     pass
#  integrate along a line
def line_weight_integral(pos1, pos2, weight_volume):

    # find all points between start and end
    inline_points = all_points_inline(pos1, pos2)
    # print(inline_points)
    # points_num = inline_points.shape[0]
    line_weight = 0
    for inline_point in inline_points:
        point_weight = weight_volume[inline_point[0],
                                    inline_point[1],
                                    inline_point[2]]
        line_weight = line_weight + point_weight

    return line_weight

#  get all lines along a point
def all_points_inline(pos1, pos2):

    d = np.diff(np.array((pos1, pos2)), axis=0)[0]
    j = np.argmax(np.abs(d))
    D = d[j]
    aD = np.abs(D)
    return pos1 + (np.outer(np.arange(aD + 1), d) + (aD >> 1)) // aD

