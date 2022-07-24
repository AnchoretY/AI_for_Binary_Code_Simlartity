'''
Author: Yhk
Date: 2022-05-27 04:43:05
LastEditors: AnchoretY
LastEditTime: 2022-06-02 07:06:26
Description: 
'''
import os
import sys
sys.path.append(os.getcwd())

import glob
import torch
import numpy
import logging
import networkx as nx
from tqdm import tqdm
from networkx.drawing.nx_agraph import read_dot,write_dot
from order_matter.bert.data_generate import block_content_parse,read_json
from order_matter.bert.vocab import WordVocab
from order_matter.config import USE_CUDA,CUDA_DEVICE

from util.log_helper import get_logger
logger = get_logger("order_matter/block_embedding.log")

def load_block_encoder(                                            
        model_path="model_file/block_embedding/bert_trained.model.ep4",
        vocab_path="data/order_matter/vocab"
    ):
    encoder = Block_Encoder(model_path,vocab_path)
    return encoder


class Block_Encoder:
    def __init__(self, model_path, vocab_path):
        logging.info("Loading Vocab{}".format(vocab_path))
        self.vocab = WordVocab.load_vocab(vocab_path)
        logging.info("Vocab Size:{} ".format(len(self.vocab)))
        self.model = torch.load(model_path)
        logging.info("Load block encoder model....")
        if USE_CUDA:
            self.model.cuda(CUDA_DEVICE)
        logging.info("Load block encoder model completed!")


    def encode(self, text,seq_len=128):
        """
            text: block中各条指令用空格拼接组成的字符串
            seq_len: block 指令组成的字符串最大长度，大于进行截断，小于则用0进行填充
        """
        segment_label = []
        sequence = []

        # segment_label = (len(text.split(' '))+2) * [1]
        sequence = self.vocab.to_seq(text) # TODO：这里是否直接加个参数即可
        sequence = [self.vocab.sos_index] + sequence + [self.vocab.eos_index]
        segment_label = len(sequence)*[1]
        
        if len(segment_label) > seq_len:
            segment_label = segment_label[:seq_len]
        else:
            segment_label = segment_label + [0]*(seq_len-len(segment_label))

        if len(sequence) > seq_len:
            sequence = sequence[:seq_len]
        else:
            sequence = sequence + [0]*(seq_len-len(sequence))
         
        segment_label = torch.LongTensor(segment_label)
        sequence = torch.LongTensor(sequence)

        segment_label = torch.unsqueeze(segment_label,0)
        sequence = torch.unsqueeze(sequence,0)

        if USE_CUDA:
            sequence = sequence.cuda(CUDA_DEVICE)
            segment_label = segment_label.cuda(CUDA_DEVICE)
        
        encoded = self.model.forward(sequence, segment_label)
        # 将每一个token的向量表示取均值作为整个block的向量表示
        result = torch.mean(encoded.detach(), dim=1)
        result = torch.squeeze(result,0)

        del encoded
        if USE_CUDA:
            if numpy:
                return result.data.cpu().numpy()
            else:
                return result.to('cpu')
        else:
            if numpy:
                return result.data.numpy()
            else:
                return result


def transform_block_embedding_graph(origin_graph,symbol_map,string_map,blcok_embedding_model):
    """
        将原始cfg生成block embedding后的cfg图
        Parameters:
            origin_graph: 原始cfg图
            symbol_map：符号与地址的映射字典
            string_map: 字符串与地址的字典
    """
    asm = nx.get_node_attributes(origin_graph,'text')   # 获取每个block包含的指令字符串
    G = nx.DiGraph()
    map_dict = {ins_address:index for index,ins_address in enumerate(asm.keys())}
    # 将每个block中的每条指令创建一个节点，block内指令顺序建立边连接
    for block_address in origin_graph.nodes():
        block_content = asm[block_address]
        _,ins_l = block_content_parse(block_content,symbol_map,string_map) 
        block_ins_string = " ".join(ins_l)
        block_embedding= list(blcok_embedding_model.encode(block_ins_string))  # 这里需要确定的是能不能进行完整的编码
        G.add_node(map_dict[block_address],embedding=block_embedding)

    # 复原边
    for edge in origin_graph.edges:
            G.add_edge(map_dict[edge[0]],map_dict[edge[1]])
    return G

def generate_block_embedding_cfg_to_file(dataset_path,generate_file_path=None):
    """
        遍历原始cfg文件目录，生成block embedding后的cfg图，存储到文件中供dataset直接读取
    """
    save_samples = 0
    file_l = []
    if not dataset_path.endswith("/"):
        dataset_path = dataset_path+"/"
    if not generate_file_path.endswith("/"):
        generate_file_path = generate_file_path+"/"
    block_encoder = load_block_encoder()
    

    # 遍历原始数据集目录中的cfg文件，获得需要转换的文件路径列表
    for program in os.listdir(dataset_path):
        program_path = os.path.join(dataset_path,program)
        for library in os.listdir(program_path):
            libraray_path = os.path.join(program_path,library)
            complie_type_l = os.listdir(libraray_path)
            for complie_type in complie_type_l:
                complie_path = os.path.join(libraray_path,complie_type)
                cfg_filename_l = [os.path.basename(cfg_file) for cfg_file in glob.glob(os.path.join(complie_path,"cfg*"))]
                for filename in cfg_filename_l:
                    file = os.path.join(complie_path,filename)
                    file_l.append(file)
    pre_path = ""
    for file in file_l:
        path = os.path.dirname(file)
        filename = os.path.basename(file)
        if path!=pre_path:
            try:
                symbol_map = read_json(os.path.join(path,"symbol_map.json"))
                string_map = read_json(os.path.join(path,"string_map.json"))
                pre_path = path
            except:
                continue
            if generate_file_path==None:
                block_embedding_graph_path = path.replace(dataset_path,dataset_path[:-1]+"_block_embedding_graph/")
            else:
                block_embedding_graph_path = path.replace(dataset_path,generate_file_path)

        block_embedding_graph_file = os.path.join(block_embedding_graph_path,filename)   
        try:
            graph = read_dot(file)
            if not is_large_func(graph):
                logging.info("Drop function {} ,because function block nums less than 5.".format(file))
                continue
        except ValueError as e:
            continue
        try:
            graph = transform_block_embedding_graph(graph,symbol_map,string_map,block_encoder)
            save_samples+=1
        except:
            logging.info("transform_block_embedding_graph function excute fail!")
            
            continue

        if not os.path.exists(block_embedding_graph_path):
            logging.info("创建目录：{}".format(block_embedding_graph_path))
            os.makedirs(block_embedding_graph_path)
        write_dot(graph,block_embedding_graph_file)
        
    logging.info("Origin cfg file nums :{}".format(len(file_l)))
    logging.info("Generate block embedding cfg nums:{}".format(save_samples))          
    return save_samples
    
def is_large_func(graph):
    if len(graph.nodes())>5:
        return True
    else:
        return False

if __name__=="__main__":
    # from networkx.drawing.nx_agraph import read_dot
    # from order_matter.bert.data_generate import read_json
    # test_cfg = "data/train/coreutils-9.1/id/x86-64_O0_clang/cfg_0x004041b0_quotearg_buffer_restyled.dot"
    # symbol_map = read_json("data/train/coreutils-9.1/id/x86-64_O0_clang/symbol_map.json")
    # string_map = read_json("data/train/coreutils-9.1/id/x86-64_O0_clang/string_map.json")
    # origin_cfg = read_dot(test_cfg)
    # block_embedding_model = load_block_encoder()

    # block_embedding_cfg = transform_block_embedding_graph(origin_cfg,symbol_map,string_map,block_embedding_model)
    # logging.info("Transform completed!")
    generate_block_embedding_cfg_to_file("data/train/","data/order_matter/train")
