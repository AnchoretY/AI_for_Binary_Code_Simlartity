'''
Author: AnchoretY
Date: 2022-05-30 04:40:21
LastEditors: AnchoretY
LastEditTime: 2022-06-02 03:04:33
'''
"""
Configuration file.
"""

VOCAB_SIZE = 5000
USE_CUDA = True
DEVICES = [0]
CUDA_DEVICE = DEVICES[0]

# data generator
MAX_SEQUENCE_LEN_IN_BLOCK = 100     # 每个指令序列的最大长度
MAX_BLOCK_SEQUENCE_LEN = 5         # 每个节点产生随机游走序列的最大个数

# bert模型参数
BERT_EMBEDDING_SIZE = 128
BERT_SEQUENCE_LEN = 128
BATCH_SIZE = 10

# 函数编码模型参数
GRAPH_EMBEDDING_SIZE = 64
EDGE_EMBEDDING_SIZE = 4
LEARNING_RATE=1e-4

