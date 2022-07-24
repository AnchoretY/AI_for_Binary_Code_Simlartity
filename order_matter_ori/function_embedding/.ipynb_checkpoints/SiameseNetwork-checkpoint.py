'''
Author: AnchoretY
Date: 2022-04-26 22:38:08
LastEditors: Yhk
LastEditTime: 2022-05-10 04:35:04
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from gmn.utils import build_GraphEmbeddingNet
from gmn.configure import get_default_config


config = get_default_config()
# 基于GNN的孪生神经网络
class SiameseNetworkGnn(nn.Module):
    def __init__(self,gnn_embedding_dim,edge_state_dim):
        super().__init__()
        self.gnn,_ = build_GraphEmbeddingNet(config,gnn_embedding_dim,edge_state_dim)

    def forward_once(self,x):
        output = self.gnn(**x)
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
#         loss_cos_con = torch.mean((1-label) * torch.div(torch.pow((1.0-cos_sim), 2), 4) +
#                                     (label) * torch.pow(cos_sim * torch.lt(cos_sim, self.margin), 2))
        loss_cos_con = torch.mean(torch.abs(label-cos_sim))
        return loss_cos_con
    
if __name__=="__main__":
    pass
    # siamese_model = SiameseNetworkGnn(128,4)
    # g1_input = [node_features, edge_features, from_idx, to_idx, graph_idx,n_graph]
    # g2_input = g1_input

    # output1,output2 = siamese_model(g1_input,g2_input)
    # loss = CosineContrastiveLoss()
    # loss(output1,output2,1)
