'''
Author: AnchoretY
Date: 2022-06-06 05:59:47
LastEditors: Yhk
LastEditTime: 2022-06-20 04:30:37
'''
import torch
import torch.nn as nn

bce, sigmoid, softmax = nn.BCELoss(), nn.Sigmoid(), nn.Softmax(dim=1)

class ASM2VEC(nn.Module):
    def __init__(self, vocab_size, function_size, embedding_size):
        super(ASM2VEC, self).__init__()
        self.embeddings   = nn.Embedding(vocab_size, embedding_size, _weight=torch.zeros(vocab_size, embedding_size)) # 中心词权重矩阵embedding层
        self.embeddings_f = nn.Embedding(function_size, 2 * embedding_size, _weight=(torch.rand(function_size, 2 * embedding_size)-0.5)/embedding_size/2) # 函数权重矩阵的embedding层
        self.embeddings_r = nn.Embedding(vocab_size, 2 * embedding_size, _weight=(torch.rand(vocab_size, 2 * embedding_size)-0.5)/embedding_size/2) # 周围词权重矩阵embedding层

    def update(self, function_size_new, vocab_size_new):
        device = self.embeddings.weight.device
        vocab_size, function_size, embedding_size = self.embeddings.num_embeddings, self.embeddings_f.num_embeddings, self.embeddings.embedding_dim
        # 如果新的词汇表大小比以前的词汇表大
        if vocab_size_new != vocab_size:
            # 增加词汇表embedding层权重，全0初始化
            weight = torch.cat([self.embeddings.weight, torch.zeros(vocab_size_new - vocab_size, embedding_size).to(device)])
            self.embeddings = nn.Embedding(vocab_size_new, embedding_size, _weight=weight)
            # 增加预测前后词汇表的embedding层权重，（0，1）之间随机初始化
            weight_r = torch.cat([self.embeddings_r.weight, ((torch.rand(vocab_size_new - vocab_size, 2 * embedding_size)-0.5)/embedding_size/2).to(device)])
            self.embeddings_r = nn.Embedding(vocab_size_new, 2 * embedding_size, _weight=weight_r)
        # 增加函数embedding层权重，（0，1）之间随机初始化
        self.embeddings_f = nn.Embedding(function_size_new, 2 * embedding_size, _weight=((torch.rand(function_size_new, 2 * embedding_size)-0.5)/embedding_size/2).to(device))

    def v(self, inp):
        """
            使用上下文指令和函数向量预测当前词的embedding
        """
        e  = self.embeddings(inp[:,1:])   # 获得操作数、操作码embedding
        v_f = self.embeddings_f(inp[:,0]) # 获取函数embedding
        v_prev = torch.cat([e[:,0], (e[:,1] + e[:,2]) / 2], dim=1)   # 前一条指令的embedding计算
        v_next = torch.cat([e[:,3], (e[:,4] + e[:,5]) / 2], dim=1)   # 后一条指令的embedding计算
        v = ((v_f + v_prev + v_next) / 3).unsqueeze(2)  # 预测目标指令的embedding
        return v

    def forward(self, inp, pos, neg):
        """
                inp: [batch_size,7],函数索引+上下文指令token对应的索引组成的tensor
                pos: [batch_size,3]，目标指令token对应的索引
                neg: [batch_size,negtive_smaples]，负采样得到的token对应的索引
        """
        device, batch_size = inp.device, inp.shape[0]
        v = self.v(inp)
        
        # negative sampling loss
        pred = torch.bmm(self.embeddings_r(torch.cat([pos, neg], dim=1)), v).squeeze()  #矩阵相乘相当于向量点积，导表两个向量之间的相似度
        label = torch.cat([torch.ones(batch_size, 3), torch.zeros(batch_size, neg.shape[1])], dim=1).to(device)
        return bce(sigmoid(pred), label)

    def predict(self, inp, pos):
        device, batch_size = inp.device, inp.shape[0]
        v = self.v(inp)
        probs = torch.bmm(self.embeddings_r(torch.arange(self.embeddings_r.num_embeddings).repeat(batch_size, 1).to(device)), v).squeeze(dim=2)
        return softmax(probs)
