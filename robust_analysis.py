import os
import glob
import pickle
import numpy as np
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

sns.set(style="darkgrid")
Colors=['red','red', 'red', 'blue','blue','blue']

# # [200]: 181210plc1p2, 170704plc1p1
# # [170]: *200311plc1p2*, 200311plc1p3, 200312plc1p1, 200312plc1p3, 200314plc1p3, *200315plc1p1*, 200316plc1p3, 181210plc1p1
# # [165]: 200309plc1p1, 200312plc1p2,
# # [160]: 200309plc1p2, 200309plc1p3, *200310plc1p2*, 200311plc1p1, 200315plc1p2, 200315plc1p3, 200316plc1p1, 200316plc1p2
# # [155]: 200314plc1p2
# # [150]: 200314plc1p1, 181210plc1p3

t_max = 150
nucleus_folder = "./ResultCell/NucleusLoc"
# get all embryos names
embryos =[os.path.join(nucleus_folder, embryo_name) for embryo_name in ["170704plc1p1"]]

with open("./ShapeUtil/number_dictionary.txt", "rb") as f:
    number_dict = pickle.load(f)

#######################################
# measure nucleus lost ratio
######################################

plt.clf()
for i in range(len(embryos)):
# For the first embryo
    embryo = embryos[i]
    ts = []
    nucleus_nums = []
    nucleus_losts = []
    for t in tqdm(range(1, t_max+1), desc="Processing {} TP".format(embryo)):

        nucleus_loc_file = os.path.join(embryo, os.path.basename(embryo)+"_"+str(t).zfill(3)+"_nucLoc"+".csv")
        pd_loc = pd.read_csv(nucleus_loc_file)
        nucleus_num = pd_loc[pd_loc.note=="mother"].shape[0] + pd_loc.note.isna().sum() + pd_loc[pd_loc.note=="lost_inner1"].shape[0] + \
                      pd_loc[pd_loc.note == "lost_inner2"].shape[0] + pd_loc[pd_loc.note=="lost_hole"].shape[0]
        nucleus_lost = pd_loc[pd_loc.note=="lost_inner1"].shape[0] + pd_loc[pd_loc.note == "lost_inner2"].shape[0] + \
                       pd_loc[pd_loc.note=="lost_hole"].shape[0]
        ts.append(t)
        nucleus_nums.append(nucleus_num)
        nucleus_losts.append(nucleus_lost)
    nucleus_stat = pd.DataFrame({
        "time": ts,
        "nucleus_number": nucleus_nums,
        "nucleus_lost": nucleus_losts
    })
    save_file = os.path.join("./ShapeUtil/RobustStat", "nucelus_lost_"+embryo.split("/")[-1]+".csv")
    nucleus_stat.to_csv(save_file, index=False)
    sns.lineplot(x="time", y="value", hue="variable", data=pd.melt(nucleus_stat, ["time"]), markers=False, dashes=[(2, 2), (2, 2)], legend=False)
plt.xlabel(r"Time point (/$1.5min$)")
plt.ylabel("Cell number (/1)")
plt.savefig(os.path.join("./ShapeUtil/RobustStat", "nucelus_lost_all.jpeg"), dpi=600)


#######################################
# cell contact significance
######################################
# get contact maps
'''
contacts = glob.glob(os.path.join("../ResultCell/test_embryo_robust/statShape", "*_Stat.txt"))
all_pd_contact = []
for contact in contacts:
    with open(contact, "rb") as f:
        all_pd_contact.append(pickle.load(f))
contact_all = pd.concat(all_pd_contact, ignore_index=True, axis=0, sort=False, join="inner")
contact_all.to_csv("test_all_contact.csv")
'''

####################################
# combine all volume and surface information
####################################

for embryo in embryos:
    # combien all volume and surface informace
    volume_stat = pd.DataFrame([], columns=[], dtype=np.float32)
    surface_stat = pd.DataFrame([], columns=[], dtype=np.float32)
    volume_lists = []
    surface_lists = []
    for t in tqdm(range(1, t_max + 1), desc="Processing {}".format(embryo.split('/')[-1])):
        nucleus_loc_file = os.path.join(embryo, os.path.basename(embryo)+"_"+str(t).zfill(3)+"_nucLoc"+".csv")
        pd_loc = pd.read_csv(nucleus_loc_file)
        cell_volume_surface = pd_loc[["nucleus_name", "volume", "surface"]]
        cell_volume_surface = cell_volume_surface.set_index("nucleus_name")
        volume_lists.append(cell_volume_surface["volume"].to_frame().T.fillna(0))
        surface_lists.append(cell_volume_surface["surface"].to_frame().T.fillna(0))
    volume_stat = pd.concat(volume_lists, keys=range(1, t_max+1), ignore_index=True, axis=0, sort=False, join="outer")
    surface_stat = pd.concat(surface_lists, keys=range(1, t_max+1), ignore_index=True, axis=0, sort=False, join="outer")
    volume_stat.to_csv(os.path.join("./ShapeUtil/RobustStat", embryo.split('/')[-1] + "_volume"+'.csv'))
    surface_stat.to_csv(os.path.join("./ShapeUtil/RobustStat", embryo.split('/')[-1] + "_surface"+'.csv'))

#=================================save for GUI======================
# for column in volume_stat.columns:

#=================================save for GUI======================


# output and draw volume
'''
surfaces = glob.glob(os.path.join("./RobustStat", "surface_*.csv"))
volumes = glob.glob(os.path.join("./RobustStat", "volume_*.csv"))

ratio_surfaces = []
ratio_volumes = []
embryo_names = []
times = []
for surface_file, volume_file in zip(surfaces, volumes):
    plt.clf()
    embryo_name = volume_file.split('/')[-1].split('_')[-1].split('.')[0]
    # load time tree file to get nucleus fir
    volume = pd.read_csv(volume_file, dtype=np.float32, index_col=0)
    surface = pd.read_csv(surface_file, dtype=np.float32, index_col=0)
    ratio_surface = surface.std().divide(surface.mean())
    ratio_volume = volume.std().divide(volume.mean())

    ratio_surfaces = ratio_surfaces + ratio_surface.tolist()
    ratio_volumes = ratio_volumes + ratio_volume.tolist()
    embryo_names = embryo_names + [embryo_name]*ratio_volume.shape[0]
    time = volume.apply(pd.DataFrame.first_valid_index, axis=0)
    times = times + time.tolist()
# get common segCell
# cell_name_collections = ratio_volumes[0].index
# for ratio_volume in ratio_volumes:
#     cell_name_collections = list(set(cell_name_collections) & set(ratio_volume.index))
# volumes with common cell names in different embryo
combine_volume_consistency = pd.DataFrame({"volume_ratio":ratio_volumes, "start_time": times, "embryo_name":embryo_names}).dropna()
combine_volume_consistency.to_csv("./RobustStat/volume_consitency.csv", index=False)

h = sns.jointplot(x="volume_ratio", y="start_time",
                  Data=combine_volume_consistency,
                  xlim=(0, 5),
                  ylim=(0, 160),
                  s=5)
h.ax_joint.set_xlabel(r'Volume consistency coefficient $\rho_c$',)
h.ax_joint.set_ylabel(r'Time points $\rho_t$',)
plt.savefig(os.path.join("./RobustStat", "volume_all_density.jpeg"), dpi=600)
'''
####################################################
# detect the distribution of segCell with large deviation
#######################################################
