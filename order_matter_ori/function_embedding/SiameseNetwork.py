'''
Author: AnchoretY
Date: 2022-04-26 22:38:08
LastEditors: AnchoretY
LastEditTime: 2022-06-28 00:10:50
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

# 基于GNN的孪生神经网络
class SiameseNetworkGnn(nn.Module):
    def __init__(self,gnn_model):
        super().__init__()
        self.gnn = gnn_model

    def forward_once(self,x):
        # output = self.gnn(**x)
        output = self.gnn(*x)
        return output

    def forward(self,input1,input2):
        """
            input1: list,[node_features, edge_features, from_idx, to_idx, graph_idx,n_graph],需保证顺序
        """
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1,output2
        
class CosineContrastiveLoss(nn.Module):
    def __init__(self, margin=0.4):
        super(CosineContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        cos_sim = F.cosine_similarity(output1, output2)
        loss_cos_con = torch.mean(torch.abs(label-cos_sim))
        return loss_cos_con
        

    
if __name__=="__main__":
    from order_matter_ori.mpnn import MPNN_Graph_Embedding
    mpnn_gnn = MPNN_Graph_Embedding(128,4)
    siamese_gnn = SiameseNetworkGnn(mpnn_gnn)