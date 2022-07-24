'''
Author: AnchoretY
Date: 2022-04-27 05:13:07
LastEditors: Yhk
LastEditTime: 2022-05-11 05:28:49
'''
import re
import os
import torch
import random
import glob
import palmtree_embedding.eval_utils as ins_embedding_utils
from torch.utils.data import Dataset,DataLoader
from networkx.drawing.nx_agraph import read_dot
from block_embedding import transform_block_embedding_graph
from gmn.utils import get_batch_input

def load_palmtree():
    palmtree = ins_embedding_utils.UsableTransformer(
                model_path="palmtree_embedding/palmtree/transformer.ep19", 
                vocab_path="palmtree_embedding/palmtree/vocab"
            )
    return palmtree
palmtree = load_palmtree()

class FunctionSimliarityDataset(Dataset):

    def __init__(self,dataset_path):
        self.dataset_path = dataset_path
        self.samples = []
        self.labels = []

        cfg_dot_file_reg = "^cfg_0x.*?\.dot$"

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
                        removed_complied_type_l = complie_type_l[::]
                        removed_complied_type_l.remove(complie_type)
                        remove_cfg_filename_l = cfg_filename_l[::]
                        remove_cfg_filename_l.remove(filename)
                        # 对于每一个cfg，生成不同编译选项下的对应cfg，作为一个正样本
                        for pos_complie_type in removed_complied_type_l:
                            postive_file = os.path.join(libraray_path,pos_complie_type,filename)
                            # TODO:临时使用，数据正常后可删除
                            if os.path.exists(postive_file):
                                
                                # 负样本为同一程序中的随机其他文件
                                negtive_file = os.path.join(complie_path,random.choice(remove_cfg_filename_l))
                                postive_sample = (file,postive_file)
                                negtive_smaple = (file,negtive_file)
                                self.samples.extend([postive_sample,negtive_smaple])
                                self.labels.extend([1,0])
                
    def __getitem__(self,index):
        
        cfg_file1,cfg_file2 = self.samples[index]
        return cfg_file1,cfg_file2,self.labels[index]

    def __len__(self):
        return len(self.samples)

def collate_fn(batch):
    graph1_l = []
    graph2_l = []
    labels = []
    for graph1,graph2,label in batch:
        try:
            graph1_l.append(graph1)
            graph2_l.append(graph2)
            labels.append(label)
        except Exception as ex:
            print(ex)
            continue
        
    graph_input1,graph_input2 = get_batch_input(graph1_l),get_batch_input(graph2_l)
    labels = torch.Tensor(labels)
    return graph_input1,graph_input2,labels



# def collate_fn(batch):
#     graph1_l = []
#     graph2_l = []
#     labels = []
#     for cfg_file1,cfg_file2,label in batch:
#         try:
#             graph1,graph2 = read_dot(cfg_file1),read_dot(cfg_file2)
#             #graph1,graph2 = transform_block_embedding_graph(palmtree,graph1),transform_block_embedding_graph(palmtree,graph2)
#             graph1_l.append(graph1)
#             graph2_l.append(graph2)
#             labels.append(label)
#         except Exception as ex:
#             print(ex)
#             print(cfg_file1,cfg_file2)
#             continue
    
#     graph_input1,graph_input2 = get_batch_input(graph1_l),get_batch_input(graph2_l)
#     labels = torch.Tensor(labels)
#     return graph_input1,graph_input2,labels


        
if __name__=="__main__":
    dataset = FunctionSimliarityDataset("data/train_data2/")
    dataloader = DataLoader(dataset,batch_size=4,collate_fn=collate_fn)
    graph_input1,graph_input2,labels = next(iter(dataloader))