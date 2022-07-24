'''
Author: Yhk
Date: 2022-05-15 22:53:43
LastEditors: Yhk
LastEditTime: 2022-07-21 10:17:57
Description: 
'''
import os
import glob
import json
import pickle
import pandas as pd
from tqdm import tqdm

def read_datapath_file(dataset_path):
    """
        读取数据集路径，获得其中包含的cfg文件列表
    """
    file_l = []
    for program in os.listdir(dataset_path):
        program_path = os.path.join(dataset_path,program)
        for library in os.listdir(program_path):
            libraray_path = os.path.join(program_path,library)
            complie_type_l = os.listdir(libraray_path)
            for complie_type in complie_type_l:
                complie_path = os.path.join(libraray_path,complie_type)
                cfg_filename_l = [os.path.basename(cfg_file) for cfg_file in glob.glob(os.path.join(complie_path,"cfg*"))]
                for filename in cfg_filename_l:
                    file = os.path.join(complie_path,filename)
                    file_l.append(file)
    return file_l

def read_json(file):
    with open(file,'r') as load_f:
         load_dict = json.load(load_f)
    return load_dict

def read_pickle(file):
    with open(file,'rb') as f:
        data = pickle.load(f)
    return data

def write_pickle(data,file):
    with open(file,'wb') as f:
        data = pickle.dump(data,f)
    

def get_function_name(file,part_nums=3):
    return file.split("/")[-1].split("_",part_nums-1)[-1]
    
def get_function_origin_info(file):
    func_name = os.path.split(file)[1].split(".")[0]
    path,complie_info = os.path.split(os.path.dirname(file))
    word_size,optimizer,complier = complie_info.split("_")
    path,library = os.path.split(path)
    _,program = os.path.split(path)
    return func_name,program,library,word_size,optimizer,complier

def get_file_list(file_path):

    file_l = []
    for program in os.listdir(file_path):
        program_path = os.path.join(file_path,program)
        for library in os.listdir(program_path):
            libraray_path = os.path.join(program_path,library)
            complie_type_l = os.listdir(libraray_path)
            for complie_type in complie_type_l:
                complie_path = os.path.join(libraray_path,complie_type)
                cfg_filename_l = [os.path.basename(cfg_file) for cfg_file in glob.glob(os.path.join(complie_path,"cfg*"))]
                for filename in cfg_filename_l:
                    file = os.path.join(complie_path,filename)
                    file_l.append(file)
    return file_l

def get_group_names(file_id,df_group):
    """
    根据文件id在df_group中查找该文件所在组的id
    """
    group_names = []
    for i,row in df_group.iterrows():
        this_group_names = row.to_list()
        if file_id in this_group_names:
            group_names =  [x for x in this_group_names if pd.isnull(x) == False]
            break
    group_names.remove(file_id)
    return group_names

def get_group_index(file_id,df_group):
    """
    根据文件id在df_group中查找该文件所在组的id
    """
    for group_id,row in df_group.iterrows():
        this_group_names = row.to_list()
        if file_id in this_group_names:
            break
    return group_id

def get_file_group_map(group_file):
    """
        Args:
            group_file: 分组文件地址
        Return:
            group_to_file_map: group_id映射到所属组包含的file_id字典
            file_to_group_map: file_id映射到所属group_id的字典
    """
    df_group = pd.read_csv(group_file,names=range(62))
    group_to_file_map = {}
    file_to_group_map = {}
    for _,row in tqdm(df_group.iterrows()):
        row_data = row.tolist()
        group_id,row_data = int(row_data[0]),row_data[1:]
        file_l = [int(x) for x in row_data if pd.isnull(x) == False]
        group_to_file_map[group_id] = file_l
        for file in file_l:
            file_to_group_map[file]=group_id
    return group_to_file_map,file_to_group_map