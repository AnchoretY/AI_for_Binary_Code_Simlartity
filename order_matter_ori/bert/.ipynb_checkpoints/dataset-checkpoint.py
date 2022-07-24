'''
Author: Yhk
Date: 2022-05-24 04:03:22
LastEditors: AnchoretY
LastEditTime: 2022-05-30 05:25:23
Description: 
'''
import os
import sys
sys.path.append(os.getcwd())

from torch.utils.data import Dataset
import tqdm
import torch
import random
import pickle as pkl

# Bert中数据文件中数据格式：每一行两句话，中间用\t隔开
class BERTDataset(Dataset):
    """

    """
    def __init__(self, cfg_corpus_path, vocab, seq_len, max_samples=-1,encoding="utf-8"):
        """

        """
        self.vocab = vocab
        self.seq_len = seq_len

        self.bb_len = 50

        self.corpus_lines = 0
        self.cfg_lines = []
        self.cfg_corpus_path = cfg_corpus_path
        self.encoding = encoding

        with open(cfg_corpus_path, "r", encoding=encoding) as f:
            # 统计输入的样本对数据行数
            for line in f:
                if max_samples==self.corpus_lines:
                    break
                self.cfg_lines.append(line[:-1].split("\t"))
                self.corpus_lines += 1

            # 获得每句话内容
            # self.cfg_lines = [line[:-1].split("\t") for line in f]
            
            if self.corpus_lines > len(self.cfg_lines):    
                self.corpus_lines = len(self.cfg_lines)
            

    def __len__(self):
        return self.corpus_lines

    # 使用索引进行内容访问
    def __getitem__(self, item):
        
        c1, c2, c_label = self.random_sent(item)

        # 任务1：MLM任务样本和标签
        c1_random, c1_mlm_label = self.random_word(c1)
        c2_random, c2_mlm_label = self.random_word(c2)

        c1_mlm = [self.vocab.sos_index] + c1_random + [self.vocab.eos_index]    # 在第一条序列化的指令后前后加入起始、结束标记索引
        c2_mlm = c2_random + [self.vocab.eos_index]                             # 在第二条序列化的指令后加入结束标记索引

        c1_mlm_label = [self.vocab.pad_index] + c1_mlm_label + [self.vocab.pad_index]
        c2_mlm_label = c2_mlm_label + [self.vocab.pad_index]

        bert_mlm_input = (c1_mlm + c2_mlm)[:self.seq_len]
        bert_mlm_label = (c1_mlm_label + c2_mlm_label)[:self.seq_len]
        
        padding = [self.vocab.pad_index for _ in range(self.seq_len - len(bert_mlm_input))]
        bert_mlm_input.extend(padding)
        bert_mlm_label.extend(padding)
        
        # 任务2：nsp任务样本和标签
        c1 = [self.vocab.sos_index] + [self.vocab.stoi.get(c, self.vocab.unk_index) for c in c1.split()] + [self.vocab.eos_index]
        c2 = [self.vocab.stoi.get(c, self.vocab.unk_index) for c in c2.split()] + [self.vocab.eos_index]
        
        bert_nsp_input = (c1 + c2)[:self.seq_len]
        cfg_padding = [self.vocab.pad_index for _ in range(self.seq_len - len(bert_nsp_input))]
        bert_nsp_input.extend(cfg_padding)

        # cfg的段标签构造
        segment_label = ([1 for _ in range(len(c1))] + [2 for _ in range(len(c2))])[:self.seq_len]
        segment_label.extend(cfg_padding)
        # 任务3：GC块所属的函数编译选项、编译级别等分类
        
        
        # 任务4：BIG两个基本块是否在一个图中
        


        output = {
                "bert_mlm_input":bert_mlm_input,
                "bert_mlm_label":bert_mlm_label,
                "bert_nsp_input": bert_nsp_input,
                "segment_label": segment_label,
                "is_next_label": c_label
                }

        return {key: torch.tensor(value) for key, value in output.items()}


    def random_word(self, sentence):
        """
            为后续mask任务构造样本。
            85%的样本保持原样，样本标签为0；另外15%中，80%做mask，10%替换成其他随机词，10%保持原样，同时标签为对应的数字；同时还将sentence转化为index序列形式。
        """
        tokens = sentence.split()
        output_label = []

        for i, token in enumerate(tokens):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = self.vocab.mask_index

                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = random.randrange(len(self.vocab))

                # 10% randomly change token to current token
                else:
                    tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)

                output_label.append(self.vocab.stoi.get(token, self.vocab.unk_index))

            else:
                tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
                output_label.append(0)
        
        return tokens, output_label


    def random_sent(self, index):
        """
            以相等的概率随机选择为两条连续的指令生成标签1，或者生成一条指令与随机指令生成标签0，cfg与dfg的指令对都做同样的操作。
            为Next Sentence Preidct任务构造样本
        """
        # 随机生成cfg标签和dfg标签
        c1, c2 = self.get_corpus_line(index)
        dice = random.random() # TODO: should throw the dice twice here. 
        if dice > 0.5:
            return c1, c2, 1
        else:
            return c1, self.get_random_line(), 0


    def get_corpus_line(self, item):
        """
            获取语料库中指定行cfg和dfg指令对，并将其切分
            Parameter item： 选择的行
            Return: cfg指令对前第一条指令，cfg语料库
        """
        return self.cfg_lines[item][0], self.cfg_lines[item][1]


    def get_random_line(self):
        """
            从cfg图中随机选择一行，返回改行的后一条指令
        """
        l = self.cfg_lines[random.randrange(len(self.cfg_lines))]
        return l[1]

if __name__=="__main__":
    from order_matter.vocab import WordVocab
    from torch.utils.data import DataLoader
    from order_matter.model.bert import BERT

    train_cfg_dataset = "data/order_matter/cfg_train.txt"
    vocab_path = "/home/yhk/github/PalmTree/data/vocab"

    # with open(train_cfg_dataset, "r", encoding="utf-8") as f1:
    #     vocab = WordVocab(f1, max_size=13000, min_freq=1)
           
    # print("VOCAB SIZE:", len(vocab))
    # vocab.save_vocab(vocab_path)


    print("Loading Vocab", vocab_path)
    vocab = WordVocab.load_vocab(vocab_path)
    print("Vocab Size: ", len(vocab))


    print("Loading Train Dataset")
    train_dataset = BERTDataset(train_cfg_dataset, vocab, seq_len=500)
    print("Sample_nums:{}".format(train_dataset.corpus_lines))