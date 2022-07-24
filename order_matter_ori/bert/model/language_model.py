'''
Author: Yhk
Date: 2022-05-26 05:30:42
LastEditors: AnchoretY
LastEditTime: 2022-07-05 22:33:57
Description: 
'''
import torch.nn as nn
import torch

from .bert import BERT

from order_matter_ori.config import OPTIMIZE_MAP

class BERTLM(nn.Module):
    """
    BERT Language Model
    Next Sentence Prediction Model + Masked Language Model
    """

    def __init__(self, bert: BERT, vocab_size):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

        super().__init__()
        self.bert = bert
        self.NSP = NextSentencePrediction(self.bert.hidden)
        self.NSP_MASK = NextSentencePrediction(self.bert.hidden)
        self.MLM = MaskedLanguageModel(self.bert.hidden, vocab_size)
        self.BIG = BlockInSameGraph(self.bert.hidden)
        self.GC = GraphClassfication(self.bert.hidden,len(OPTIMIZE_MAP))
        

    def forward(self, c,c_mask,c_segment_label,big_input,big_segment_label,gc_input,gc_segment_label):
        c = self.bert(c, c_segment_label)
        c_mask = self.bert(c_mask, c_segment_label)
        big_input = self.bert(big_input,big_segment_label)
        gc_input = self.bert(gc_input,gc_segment_label)
        return self.MLM(c_mask),self.NSP(c),self.NSP_MASK(c_mask),self.BIG(big_input),self.GC(gc_input)


class NextSentencePrediction(nn.Module):
    """
    From NSP task, now used for DUP and CWP
    """
    def __init__(self, hidden):
        """
        :param hidden: BERT model output size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, 2)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x[:, 0]))

class BlockInSameGraph(nn.Module):
    def __init__(self, hidden):
        """
        :param hidden: BERT model output size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, 2)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x[:, 0]))

class MaskedLanguageModel(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, vocab_size):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))

class GraphClassfication(nn.Module):
    """
        预测基本块属于的图属于的优化选项
    """

    def __init__(self, hidden, graph_type_nums):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, graph_type_nums)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x[:, 0]))
