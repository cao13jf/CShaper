'''
This llibrary defines all structures that will be used in the shape analysis
'''

import os
import glob
import pickle
import pandas as pd
from treelib import Tree, Node


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
