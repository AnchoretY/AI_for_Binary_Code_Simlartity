'''
Author: AnchoretY
Date: 2022-06-30 04:50:31
LastEditors: AnchoretY
LastEditTime: 2022-07-11 22:43:16
'''

from torch import nn

from dgl import function as fn
from dgllife.model.readout.mlp_readout import MLPNodeReadout

class MPNNConv(nn.Module):
    def __init__(self,
                 in_feats,
                 aggregator_type='sum',
                 num_step_message_passing = 6
                ):
        super(MPNNConv, self).__init__()
        self._in_feats = in_feats
        self.num_step_message_passing = num_step_message_passing
        
        if aggregator_type == 'sum':
            self.reducer = fn.sum
        elif aggregator_type == 'mean':
            self.reducer = fn.mean
        elif aggregator_type == 'max':
            self.reducer = fn.max
        else:
            raise KeyError('Aggregator type {} not recognized: '.format(aggregator_type))
        self._aggre_type = aggregator_type
        
        self.MLP = nn.Sequential(
            nn.Linear(in_feats, in_feats),
            nn.ReLU(),
            nn.Linear(in_feats, in_feats)
        )
        
        # 更新函数
        self.gru = nn.GRU(in_feats, in_feats)


    def forward(self, graph, node_feats):

        with graph.local_scope():
            for _ in range(self.num_step_message_passing):
                # (n, d_in)
                graph.srcdata['h'] = self.MLP(node_feats)
                # 消息函数+聚合函数(n, d_in, d_out)
                graph.update_all(fn.copy_u('h', 'm'), self.reducer('m', 'neigh'))
                message = graph.dstdata['neigh'] # (n, d_out)
                # 节点状态更新
                node_feats,_ = self.gru(message.unsqueeze(0),node_feats.unsqueeze(0))
                node_feats = node_feats.squeeze(0)
                
            return node_feats


class MPNN_Graph_Embedding(nn.Module):
    """MPNN for regression and classification on graphs.
    """
    def __init__(self,
                 node_in_feats,
                 graph_feats,
                 num_step_message_passing=6,
                ):
        super(MPNN_Graph_Embedding, self).__init__()

        self.node_update = MPNNConv(
            in_feats = node_in_feats,
            aggregator_type='sum',
            num_step_message_passing = num_step_message_passing
        )
        
        self.readout = MLPNodeReadout(node_feats=node_in_feats,
                                   hidden_feats=node_in_feats,
                                   graph_feats=graph_feats)
    

    def forward(self, g, node_feats):
        node_feats = self.node_update(g, node_feats)
        graph_feats = self.readout(g, node_feats)
        return graph_feats