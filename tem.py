
from itertools import combinations
import math
import numpy as np
import nibabel as nib
from scipy.spatial import Delaunay
from scipy import ndimage
from scipy.ndimage.morphology import distance_transform_edt
import matplotlib.pyplot as plt
from skimage.morphology import h_maxima
from Util.post_lib import construct_weighted_graph, line_weight_integral, generate_graph_model
from Util.segmentation_post_process import save_nii


#==========================================================================
#            Prepare GUI Nucleus annotations
#==========================================================================

# bin_image = nib.load("ResultCell/BinaryMemb/181210plc1p2/181210plc1p2_021_segMemb.nii.gz").get_fdata().transpose() == 0
#
# volume_shape = bin_image.shape
# bin_cell = ndimage.morphology.binary_opening(bin_image).astype(np.float)
# bin_memb = bin_cell == 0
# bin_cell_edt = ndimage.morphology.distance_transform_edt(bin_cell)
#
# # get local maximum SegMemb
# local_maxima_mask = h_maxima(bin_cell_edt, 2)
# [maxima_x, maxima_y, maxima_z] = np.nonzero(local_maxima_mask)
# #  find boundary points to force large weight
# x0 = np.where(maxima_x == 0)[0]; x1 = np.where(maxima_x == volume_shape[0] - 1)[0]
# y0 = np.where(maxima_y == 0)[0]; y1 = np.where(maxima_y == volume_shape[1] - 1)[0]
# z0 = np.where(maxima_z == 0)[0]; z1 = np.where(maxima_z == volume_shape[2] - 1)[0]
# b_indx = np.concatenate((x0, y0, z0, x1, y1, z1), axis=None).tolist()
#
# point_list = np.stack((maxima_x, maxima_y, maxima_z), axis=1)
# tri_of_max = Delaunay(point_list)
# triangle_list = tri_of_max.simplices
# edge_list = []
# for i in range(triangle_list.shape[0]):
#     for combine_pairs in combinations(triangle_list[i].tolist(), r=2):
#         edge_list.append([combine_pairs[0], combine_pairs[1]])
# # add edges for all boundary points
# for i in range(len(b_indx)):
#     for j in range(i, len(b_indx)):
#         one_point = b_indx[i]
#         another_point = b_indx[j]
#         if ([one_point, another_point] in edge_list) or ([another_point, one_point] in edge_list):
#             continue
#         edge_list.append([one_point, another_point])
#
# weights_volume = bin_memb * 10000  # construct weights volume for graph
# edge_weight_list = []
# for one_edge in edge_list:
#     start_x0 = point_list[one_edge[0]]
#     end_x1   = point_list[one_edge[1]]
#     if (one_edge[0] in b_indx) and (one_edge[1] in b_indx):
#         edge_weight = 0  # All edges between boundary points are set as zero
#     elif (one_edge[0] in b_indx) or (one_edge[1] in b_indx):
#         edge_weight = 10000 * 10
#     else:
#         edge_weight = line_weight_integral(start_x0, end_x1, weights_volume)
#
#     edge_weight_list.append(edge_weight)
#
# nii_graph = generate_graph_model(point_list, edge_list, edge_weight_list, bin_memb)
# save_nii(nii_graph, './tem.nii.gz')

#==========================================================================
#            Prepare GUI Nucleus annotations
#==========================================================================

slice_num = 70
discrete_num = 8
raw_memb = nib.load("Data/MembTraining/170704plc1p2/RawMemb/170704plc1p2_017_rawMemb.nii.gz").get_fdata()
seg_memb = nib.load("Data/MembTraining/170704plc1p2/SegMemb/170704plc1p2_017_segMemb.nii.gz").get_fdata()
seg_cell = nib.load("Data/MembTraining/170704plc1p2/SegCell/170704plc1p2_017_segCell.nii.gz").get_fdata()

background = np.logical_and(seg_memb==0, seg_cell==0)[:, :, slice_num]
slice_memb = seg_memb[:, :, slice_num]
distance_memb = distance_transform_edt(slice_memb==0)

# show raw membrane
plt.imshow(raw_memb[:, :, slice_num], cmap="gray"); plt.axis('off')
plt.savefig("./ResultTem/MethodsDistanceCompare/raw.jpeg", dpi=600)

# show membrane segmentation
plt.imshow(slice_memb, cmap="gray"); plt.axis('off')
plt.savefig("./ResultTem/MethodsDistanceCompare/membSeg.jpeg", dpi=600)


# deep watershed distance
deep_distance = distance_memb.copy()
deep_distance[background] = 0
bins = np.linspace(start=deep_distance.min(), stop=deep_distance.max()+0.001, num=discrete_num)
deep_watershed_dis =  np.digitize(deep_distance, bins)
plt.imshow(deep_watershed_dis); plt.axis('off')
plt.savefig("./ResultTem/MethodsDistanceCompare/deep_watershed_distance.jpeg", dpi=600)


# single cell distance
single_cell_distance = distance_memb.copy()
single_cell_distance[background] = 0
plt.imshow(single_cell_distance); plt.axis('off')
plt.savefig("./ResultTem/MethodsDistanceCompare/single_cell_distance.jpeg", dpi=600)

# CShaper distance
threshold = 20
cshaper_distance = distance_memb.copy()
cshaper_distance[cshaper_distance > threshold] = threshold
bins = (np.linspace(0, 1, discrete_num))**2
bins = bins / bins.max() * threshold
cshaper_distance = np.digitize(cshaper_distance, bins)
plt.imshow(cshaper_distance); plt.axis('off')
plt.savefig("./ResultTem/MethodsDistanceCompare/cshaper_distance.jpeg", dpi=600)
