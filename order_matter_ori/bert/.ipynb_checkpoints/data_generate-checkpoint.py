'''
Author: Yhk
Date: 2022-05-25 00:14:03
LastEditors: AnchoretY
LastEditTime: 2022-05-31 23:15:49
Description: 
'''
'''
Author: Yhk
Date: 2022-05-24 04:05:41
LastEditors: Yhk
LastEditTime: 2022-05-25 00:13:45
Description: 生成语料库
'''
'''
Author: Yhk
Date: 2022-04-18 03:29:14
LastEditors: Yhk
LastEditTime: 2022-05-18 01:08:41
Description: 
'''
import os
import sys
sys.path.append(os.getcwd()) # 将当前文件路径加入项目目录

import re
import json
import random
import networkx as nx
from tqdm import tqdm
from networkx.drawing.nx_agraph import read_dot

from util.common_helper import read_json
from order_matter.config import MAX_SEQUENCE_LEN_IN_BLOCK,MAX_BLOCK_SEQUENCE_LEN

def block_content_parse(block_content,symbol_map, string_map):
    """
        dot格式的block内指令进行解析，转换成指令地址列表与指令字符串列表
        Parameter
            block_content: anngr dot cfg一个基本块内的内容
    """
    # 分行
    lines = block_content.strip().split("\n")
    ins_address_l,ins_l = [],[]
    # 每行切分指令地址与指令
    for l in lines:
        ins_address,ins = l.split(":",1)
        ins = parse_instruction(ins.strip(),symbol_map,string_map)
        ins_address_l.append(ins_address.strip())
        ins_l.append(ins)
    return ins_address_l,ins_l
    

def parse_instruction(ins, symbol_map, string_map):
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
                if int(symbols[j], 16) in symbol_map:
                    symbols[j] = "symbol" # function names
                elif int(symbols[j], 16) in string_map:
                    symbols[j] = "string" # constant strings
                else:
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
    """
    sequence = []
    for n in g:
        if n != -1 and g.nodes[n]['text'] != None:
            s = []
            l = 0
            s.append(g.nodes[n]['text'])
            cur = n
            while l < length:
                nbs = list(g.successors(cur))
                if len(nbs):
                    cur = random.choice(nbs)
                    s.append(g.nodes[cur]['text'])
                    l += 1
                else:
                    break
            sequence.append(s)
            
        # 如果指令序列数量达到MAX_BLOCK_SEQUENCE_LEN条，提前停止图的遍历
        if len(sequence) > MAX_BLOCK_SEQUENCE_LEN:
            return sequence[:MAX_BLOCK_SEQUENCE_LEN]
    return sequence

def get_order_matter_cfg(cfg_data_dict,symbol_map,string_map):

    G = nx.DiGraph()

    for node in cfg_data_dict['node']:
        block_addr = node["addr"]
        block_content = ""
        block_ins_string = " ".join([parse_instruction(ins['text'],symbol_map,string_map) for ins in node["text"]])
        G.add_node(block_addr,text=block_ins_string)


    # 将原有的块间边在新图中加入
    for edge in cfg_data_dict["edge"]:
        G.add_edge(edge["from"],edge["to"])
    return G
    

def process_cfg(cfg_file,symbol_map,string_map):
    cfg_data_dict = read_json(cfg_file)
    graph = get_order_matter_cfg(cfg_data_dict,symbol_map,string_map)
    sequence = random_walk(graph, MAX_SEQUENCE_LEN_IN_BLOCK)
    return sequence
    


import glob
if __name__=="__main__":
    
    cfg_file_path = "data/train_json/"
    binary_file_list = os.listdir(cfg_file_path)
    binary_file_cfg_dict = {}       # 各个binary文件中包含的全部cfg文件位置字典
    sequence = []                   # 保存指令随机游走序列

    for program in os.listdir(cfg_file_path):
        program_path = os.path.join(cfg_file_path,program)
        for library in os.listdir(program_path):
            libraray_path = os.path.join(program_path,library)
            complie_type_l = os.listdir(libraray_path)
            for complie_type in complie_type_l:
                complie_path = os.path.join(libraray_path,complie_type)
                cfg_filename_l = [os.path.basename(cfg_file) for cfg_file in glob.glob(os.path.join(complie_path,"cfg*"))]
                try:
                    string_map = read_json(os.path.join(complie_path,"string_map.json"))
                    symbol_map = read_json(os.path.join(complie_path,"symbol_map.json"))
                except:
                    continue
                for filename in cfg_filename_l:
                    cfg_file = os.path.join(complie_path,filename)
                    this_sequence = process_cfg(cfg_file,symbol_map,string_map)
                    sequence = sequence+this_sequence

    # 存储指令随机游走序列,两个指令一行
    with open('data/order_matter/cfg_train.txt', 'a') as w:
        for s in sequence:
            if len(s) >= 2:
                    for idx in range(1, len(s)):
                        w.write(s[idx-1] +'\t' + s[idx] + '\n')        







