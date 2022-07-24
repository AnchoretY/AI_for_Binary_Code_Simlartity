
'''
Author: Yhk
Date: 2022-05-24 04:05:41
LastEditors: Yhk
LastEditTime: 2022-07-21 10:42:10
Description: 生成语料库
'''

from cgi import test
import os
from pickletools import optimize
import sys

from numpy import block, save
sys.path.append(os.getcwd()) # 将当前文件路径加入项目目录

import re
import json
import random
import logging
import networkx as nx
from tqdm import tqdm
from tqdm.gui import tqdm as tqdm_gui
tqdm.pandas(ncols=100)

from networkx.drawing.nx_agraph import read_dot

from util.log_helper import get_logger


from util.common_helper import read_json,write_pickle
from order_matter_ori.config import MAX_SEQUENCE_LEN_IN_BLOCK,MAX_BLOCK_SEQUENCE_LEN


def parse_instruction(ins):
    """
        将指令的操作数中的地址进行泛化解析，地址在符号表泛化为symbol字符串，在字符串映射表则映射为string字符串，都不在则映射为address字符串
        Parameter
            ins: 待解析的指令
            symbol_map: 地址到符号名的映射
            string_map: 起始地址到字符串内容的映射
    """
    ins = re.sub('\s+', ', ', ins, 1)
    parts = ins.split(', ')
    operand = []
    if len(parts) > 1:
        operand = parts[1:]
    for i in range(len(operand)):
        symbols = re.split('([0-9A-Za-z]+)', operand[i])
        for j in range(len(symbols)):
            if symbols[j][:2] == '0x' and len(symbols[j]) >= 6:
                symbols[j] = "address" # addresses 
        operand[i] = ' '.join(symbols)
    opcode = parts[0]
    return ' '.join([opcode]+operand)


def random_walk(g,length):
    """
        在图上随机游走，生成指定长度的指令序列集合；默认上线5000个指令序列
        Parameter:
            g: 函数图
            length： 生成随机游走序列的长度
            symbol_map: 地址到符号名的映射
            string_map: 起始地址到字符串内容的映射
        Return 
            sequence: 基本块指令序列（block_nums,各个基本块的包含1~length条指令）
    """
    sequence = []
    # 遍历图中每个节点
    for n in g:
        if n != -1 and g.nodes[n]['text'] != None:
            s = []
            l = 0
            s.append(g.nodes[n]['text'])
            cur = n
            # 单个基本块中指令如果大于length，那么只取前length条指令内容，构成一个基本块的指令序列表示
            while l < length:
                nbs = list(g.successors(cur))
                if len(nbs):
                    cur = random.choice(nbs)
                    s.append(g.nodes[cur]['text'])
                    l += 1
                else:
                    break
            sequence.append(s)
            
        # 如果基本块指令序列数量达到MAX_BLOCK_SEQUENCE_LEN条，提前停止图的遍历
        if len(sequence) > MAX_BLOCK_SEQUENCE_LEN:
            return sequence[:MAX_BLOCK_SEQUENCE_LEN]
    return sequence

def get_order_matter_cfg(cfg_data_dict):

    G = nx.DiGraph()
    for node in cfg_data_dict['node']:
        block_addr = node["addr"]
        block_ins_string = " ".join([parse_instruction(ins['text']) for ins in node["text"]])
        G.add_node(block_addr,text=block_ins_string)

    # 将原有的块间边在新图中加入
    for edge in cfg_data_dict["edge"]:
        G.add_edge(edge["from"],edge["to"])
    return G
    

def process_cfg(cfg_data_dict):
    graph = get_order_matter_cfg(cfg_data_dict)
    sequence = random_walk(graph, MAX_SEQUENCE_LEN_IN_BLOCK)
    return sequence
    
def generate_data(df,output_file="data/order_matter/cfg_train.txt"):

    def write_data(cfg_dict,f):
        sequence = process_cfg(cfg_dict)
        for s in sequence:
            if len(s) >= 2:
                for idx in range(1, len(s)):
                    f.write(s[idx-1] +'\t' + s[idx] + '\n') 

    with open(output_file, 'w') as f: 
        f.truncate()   #清空文件                       
        df.progress_apply(lambda x:write_data(x.to_dict(),f),axis=1)
    


def generate_data_append(df,output_file="data/order_matter/cfg_train_append.pkl"):
    def get_random_block_content_l(cfg_dict):     
        block_content_l = []
        for node in cfg_dict['node']:
            block_ins_content = " ".join([parse_instruction(ins['text']) for ins in node["text"]])
            block_content_l.append(block_ins_content)
        if len(block_content_l)>10:
            return random.sample(block_content_l,10)
        optimize = cfg_dict["arch"]+"_"+cfg_dict["compiler"]+"_"+cfg_dict["opti"]
        data_l.append([block_content_l,optimize])
        return block_content_l

    data_l = []
    df.progress_apply(lambda x:get_random_block_content_l(x.to_dict()),axis=1)
    write_pickle(data_l,output_file)

def generate_optimize_map(df,save_data="data/order_matter/optimize_map.pkl"):
    df["optimizer"] = df.apply(lambda x:x.arch+"_"+x.compiler+"_"+x.opti,axis=1)
    optimize_map = {opt:i for i,opt in enumerate(df["optimizer"].drop_duplicates().tolist())}
    write_pickle(optimize_map,save_data)






if __name__=="__main__":
    import pandas as pd
    from tqdm import tqdm
    from tqdm.gui import tqdm as tqdm_gui
    logger = get_logger("bert_data_generate.log")

    #df = pd.read_json("data/train_func.json",orient='records',lines=True,nrows=100)
    df = pd.read_json("../../data/Function_similarity/all_language/train_func.json",orient='records',lines=True)

    generate_data(df,output_file="data/order_matter/all_language_22w/bert_corpus/cfg_train.txt")
    generate_data_append(df,output_file="data/order_matter/all_language_22w/bert_corpus/cfg_train_append.pkl")
    generate_optimize_map(df,save_data="data/order_matter/all_language_22w/bert_corpus/optimize_map.pkl",)





