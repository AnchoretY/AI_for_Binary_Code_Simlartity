'''
Author: AnchoretY
Date: 2022-04-27 05:13:07
LastEditors: AnchoretY
LastEditTime: 2022-07-14 03:33:40
'''
import os
import sys
sys.path.append(os.getcwd())

import dgl
import random
import torch
import glob
import pandas as pd
from tqdm import tqdm
from dgl.data import DGLDataset
from util.common_helper import read_pickle,get_function_name,get_file_group_map


class FunctionSimliarityDataset(DGLDataset):
    
    def __init__(self,dataset_path,group_file,max_sample_nums=5000000):
        """
            dataset_path:数据集地址
            use_block_embedding_cfg: 数据集是否为block_embedding后的cfg图
        """
        self.dataset_path = dataset_path
        self.group_file = group_file
        self.samples_file_path = [] # 样本对文件路径列表
        self.samples = []           # 样本对读取到的graph列表（有的文件对路径读取到cfg会报错）
        self.file_pair_labels = []  # 样本对文件路径标签
        self.labels = []            # 样本对读取到graph列表
        self.max_sample_nums = max_sample_nums
        
        # 获取样本文件对和对应的标签
        self.get_sample_files(dataset_path,group_file)
        
        print("Sample file nums:{}".format(len(self.samples_file_path)))
        print("Start load sample cfg...")
        for i,file_pair in tqdm(enumerate(self.samples_file_path)):
            try:
                cfg_file1,cfg_file2 = file_pair                
                graph1 = self.get_graph(cfg_file1)
                graph2 = self.get_graph(cfg_file2)
                
                self.samples.append([graph1,graph2])
                self.labels.append(self.file_pair_labels[i])
                if len(self.samples)==self.max_sample_nums:
                    return 
            except Exception as ex:
                print(ex)
                print(cfg_file1,cfg_file2)
        print("Finished load sample cfg!")

    def get_graph(self,cfg_file):       
        graph =read_pickle(cfg_file)
        graph = dgl.from_networkx(graph,node_attrs=['embedding'])
        graph.edata["embedding"] = torch.ones(graph.num_edges(),4)
        return graph
    
    def get_sample_files(self,dataset_path,group_file):
        """
            获取正负样本文件对以及对应的标签
        """
        filename_l = os.listdir(dataset_path)
        
        group_to_file_map,file_to_group_map = get_file_group_map(group_file)
        group_l = list(group_to_file_map.keys())

        for filename in tqdm(filename_l):

            file = os.path.join(dataset_path,filename)
            try:
                group_id = file_to_group_map[int(filename)]
            except KeyError:
                continue
            group_file_l = group_to_file_map[group_id][::]
            group_file_l.remove(int(filename))

            if group_file_l==[]:
                continue
            # 对于每一个cfg，生成不同编译选项下的对应cfg，作为一个正样本
            while True:
                postive_filename = random.choice(group_file_l)
                postive_file = os.path.join(dataset_path,str(postive_filename))
                if os.path.exists(postive_file):
                    break

            while True:
                negtive_file_group_id = random.choice(group_l)
                if negtive_file_group_id!=group_id:
                    break

            negtive_filename = random.choice(group_to_file_map[negtive_file_group_id])
            negtive_file = os.path.join(dataset_path,str(negtive_filename))

            postive_sample = (file,postive_file)
            negtive_smaple = (file,negtive_file)
            self.samples_file_path.extend([postive_sample,negtive_smaple])
            self.file_pair_labels.extend([1,0])
    

    def __getitem__(self,index):
        return self.samples[index][0],self.samples[index][1],self.labels[index]

    def __len__(self):
        return len(self.samples)



def collate_fn(batch):        
    labels = torch.Tensor([e[2] for e in batch])
    
    
    graphs1 = dgl.batch([e[0] for e in batch])   #会形成一个大的图，图中包含多个无关联的小图
    nodes_feats1 = graphs1.ndata["embedding"]
    edge_feats1 = graphs1.edata["embedding"]
    
    graphs2 = dgl.batch([e[1] for e in batch])   #会形成一个大的图，图中包含多个无关联的小图
    nodes_feats2 = graphs2.ndata["embedding"]
    edge_feats2 = graphs2.edata["embedding"]
    
    
    return graphs1,nodes_feats1,edge_feats1,graphs2,nodes_feats2,edge_feats2,labels

if __name__=="__main__":
    import numpy as np
    from dgl.dataloading import GraphDataLoader
    group_file = "data/trainset_group.csv"
    dataset = FunctionSimliarityDataset("data/order_matter/train_json_block_embedding_graph/",group_file=group_file)
    dataloader = GraphDataLoader(dataset,batch_size=5,shuffle=True,collate_fn=collate_fn)