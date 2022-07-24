'''
Author: Yhk
Date: 2022-05-15 22:53:43
LastEditors: Yhk
LastEditTime: 2022-05-15 23:24:10
Description: 
'''
import os
import glob
import json

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
