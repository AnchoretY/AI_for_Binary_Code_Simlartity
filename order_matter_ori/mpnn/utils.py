'''
Author: Yhk
Date: 2022-04-24 23:16:48
LastEditors: AnchoretY
LastEditTime: 2022-06-28 23:56:38
Description: 
'''
import dgl
import torch
import numpy as np
import networkx as nx

def get_batch_input(graph_l):
    """
        批量图转化成输入gnn模型标准格式
        Args:
            graph_l: list,一批cfg
    """
    
    graphs = dgl.batch([e for e in graph_l])   #会形成一个大的图，图中包含多个无关联的小图
    nodes_feats1 = graphs.ndata["embedding"]
    edge_feats1 = graphs.edata["embedding"]
    
    return graphs,nodes_feats1,edge_feats1