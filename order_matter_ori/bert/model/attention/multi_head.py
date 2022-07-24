'''
Author: Yhk
Date: 2022-05-18 03:54:25
LastEditors: Yhk
LastEditTime: 2022-05-18 03:54:25
Description: 
'''
import torch.nn as nn
from .single import Attention


class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        """
        :param h: 头数
        :param d_model: 隐藏层维度
        """
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        # 1) 将query、key、value通过一个线性层输出的结果转成（batch_size,head_nums,seq_len,d_model//head_nums）
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) 应用注意力层在全部通道上
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) 调整各个注意力通道输出的结果（整合各个通道的注意力层输出重新合并成为一个向量）
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)
