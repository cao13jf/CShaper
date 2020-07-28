
# import dependency library
import sys
from tqdm import tqdm
import multiprocessing as mp
from skimage.morphology import watershed
from scipy.stats import mode

# import user defined library
from Util.post_lib import *
from Util.nucleus_lib import *
from Util.parse_config import parse_config


def post_process(config):

    print('Begin postprocessing')
    config_segdata = config['segdata']
    membseg_path = config_segdata['membseg_path']

    # get all files under the path
    if config_segdata.get('embryos', None) is not None:
        embryo_names = config_segdata['embryos']
    else:
        embryo_names = os.listdir(membseg_path)
        embryo_names = [embryo_name for embryo_name in embryo_names if 'plc' in embryo_name.lower()]
    membseg_path = config_segdata['membseg_path']
    all_cell_intensity = []
    parameters = []
    for one_embryo in embryo_names:
        file_names = glob.glob(os.path.join(membseg_path, one_embryo, '*.nii.gz'))
        for file_name in file_names:
            parameters.append([one_embryo,file_name, config])

    mpPool = mp.Pool(mp.cpu_count()-1)
    for _ in tqdm(mpPool.imap_unordered(run_post, parameters), total=len(parameters)):
        pass


def run_post(para):
    one_embryo = para[0]
    file_name = para[1]
    config = para[2]
    config_segdata = config['segdata']
    config_debug = config['debug']
    if config_debug.get('debug_mode', False):
        debug_folder = './ResultTem'
        # print('\nIntermediate result is saved under *./ResultTem*')
    config_result = config['result']
    tp_str = os.path.basename(file_name).split('.')[0].split("_")[1]
    py_anistropic_image0 = nib.load(file_name).get_fdata()

    # Check input is binary image
    labels = np.unique(py_anistropic_image0)
    assert len(labels)==2, 'Input of post-processing should be binary'
    cell_bin_image = np.transpose(py_anistropic_image0, [2,1,0])

    cell_bin_image = (cell_bin_image == 0).astype(np.uint8)  # change it into cell based image (binary)
    memb_bin = (cell_bin_image == 0).astype(np.uint8)  # Get binary membrane image


    #===========================================================
    #               Construct weighted graphload

    #===========================================================
    point_list, edge_list, edge_weight_list = construct_weighted_graph(cell_bin_image, local_max_h=1)
    if config_debug.get('save_graph_model', False):
        nii_graph = generate_graph_model(point_list, edge_list, edge_weight_list, cell_bin_image)
        save_nii(nii_graph, os.path.join(debug_folder, 'Graph', one_embryo, 'graph'+tp_str+'.nii.gz'))


    #===========================================================
    #               CLuster points based on their connections
    #===========================================================
    # delete all edges that come cross the membrane
    valid_edge_list = [edge_list[i] for i in range(len(edge_weight_list)) if edge_weight_list[i] < 10]
    point_tomerge_list0 = []
    for one_edge in valid_edge_list:
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

    # Combine all clusters that have shared vertexs
    cluster_tem1 = point_tomerge_list0 * 1
    cluster_tem2 = point_tomerge_list0 * 1
    point_tomerge_list = []
    merged_cluster = []
    while len(cluster_tem1):
        delete_index = []
        cluster_in1 = cluster_tem1.pop()
        if cluster_in1 in merged_cluster:
            continue
        cluster_final = set(cluster_in1)
        for cluster_in2 in cluster_tem2:
            tem_final = set(cluster_final).intersection(cluster_in2)
            if len(tem_final):
                merged_cluster.append(cluster_in2)
                cluster_final = set().union(cluster_final, cluster_in2)
        point_tomerge_list.append(list(cluster_final))


    # ===========================================================
    #               Seeded watershed segmentation
    #===========================================================
    marker_volume0 = np.zeros_like(cell_bin_image, dtype=np.uint8)
    tem_point_list = np.transpose(np.array(point_list), [1,0]).tolist()
    marker_volume0[tem_point_list[0], tem_point_list[1], tem_point_list[2]] = 1
    struc_el1 = np.ones((3, 3, 3), dtype=bool)
    marker_volume1 = ndimage.morphology.binary_dilation(marker_volume0, structure=struc_el1)
    marker_volume = ndimage.label(marker_volume1)[0]
    # EDT on mmembrane-based image
    memb_edt = ndimage.morphology.distance_transform_edt(cell_bin_image>0)
    memb_edt_reversed = memb_edt.max() - memb_edt
    # Implement watershed segmentation
    watershed_seg = watershed(memb_edt_reversed, marker_volume.astype(np.uint16), watershed_line=True)
    if config_debug.get('save_init_watershed', False):
        save_nii(watershed_seg, os.path.join(debug_folder, 'Watershed', one_embryo, 'watershed'+tp_str+'.nii.gz'))


    # ===========================================================
    #  Deal with oversegmentation based on clutered local maximum
    # ===========================================================
    merged_seg = watershed_seg.copy()
    for one_merged_points in point_tomerge_list:
        first_point = point_list[one_merged_points[0]]
        one_label = watershed_seg[first_point[0], first_point[1], first_point[2]]
        for other_point in one_merged_points[1:]:
            point_location = point_list[other_point]
            new_label = watershed_seg[point_location[0], point_location[1], point_location[2]]
            merged_seg[watershed_seg == new_label] = one_label
        one_mask = merged_seg == one_label
        one_mask_closed = ndimage.binary_closing(one_mask)
        merged_seg[one_mask_closed!=0] = one_label
    # Set background as 0
    background_label = mode(merged_seg, axis=None)[0][0]
    merged_seg[merged_seg==background_label] = 0
    merged_seg = set_boundary_zero(merged_seg)
    if config_debug.get('save_merged_seg', False):
        save_nii(merged_seg, os.path.join(debug_folder, 'Mergedseg', one_embryo, 'mergedseg' + "membT" + tp_str + '.nii.gz'))


    # ===========================================================
    #  Filter empty segmented region (without nucleus inside) with nucleus information
    # ===========================================================
    if config_result['nucleus_filter']:
        nucleus_seg = nib.load(os.path.join(config_segdata['nucleus_data_root'], one_embryo, "SegNuc", one_embryo+"_"+tp_str+"_segNuc.nii.gz"))
        nucleus_seg = nucleus_seg.get_fdata().transpose([2, 1, 0])
        merged_seg, holes = cell_prob_with_nucleus(merged_seg, nucleus_seg) #TODO: some nucleus are lost in the nucleus stack, so acetree is used to filter gaps when naming each segmented region
        if holes.sum() != 0:
            save_nii(holes,os.path.join(config_result['postseg_folder'], one_embryo+"Cavity", one_embryo + "_" + tp_str.zfill(3) + "_segCavity.nii.gz"))

    # ===========================================================
    #  Seperate membrane region from the segmentation result
    # ===========================================================
    cell_without_memb = np.copy(merged_seg)
    cell_without_memb[memb_bin != 0 ] = 0
    if config_debug.get('save_cell_nomemb', False):
        save_nii(cell_without_memb, os.path.join(debug_folder, 'NoMembCellseg', one_embryo, 'noMembCellseg' + tp_str + '.nii.gz'))


    # ===========================================================
    #  Save final result
    # ===========================================================
    save_folder = config_result['postseg_folder']
    if config_result.get('save_cell_withmemb', False):
        save_nii(merged_seg, os.path.join(save_folder, one_embryo, one_embryo + "_" + tp_str.zfill(3) + "_segCell.nii.gz"))
    elif config_result.get('save_cell_nomemb', False):
        save_nii(cell_without_memb, os.path.join(save_folder, one_embryo, one_embryo + "_" + tp_str.zfill(3) + "_segCell.nii.gz"))
    else:
        raise ValueError('No FINAL result to be saved!')

    print('Finished: ', one_embryo, '--->', tp_str)


def save_nii(img, nii_name):
    nii_folder = os.path.dirname(nii_name)
    if not os.path.isdir(nii_folder):
        os.makedirs(nii_folder)
    img = nib.Nifti1Image(np.transpose(img, [2, 1, 0]), np.eye(4))
    nib.save(img, nii_name)


if __name__ == '__main__':
    if (len(sys.argv) != 2):
        raise Exception("Invalid number of input parameters")
    config_file = str(sys.argv[1])
    assert (os.path.isfile(config_file))
    config = parse_config(config_file)
    post_process(config)


