'''
Author: Yhk
Date: 2022-05-27 04:43:05
LastEditors: AnchoretY
LastEditTime: 2022-07-21 05:20:22
Description: 
'''
import os
import sys
sys.path.append(os.getcwd())


import torch
import numpy
import logging
import networkx as nx
import pandas as pd
from tqdm import tqdm
from tqdm.gui import tqdm as tqdm_gui
from joblib import Parallel,delayed
tqdm.pandas(ncols=100)
from networkx.drawing.nx_agraph import read_dot,write_dot



from util.common_helper import read_json,write_pickle,read_pickle
from order_matter_ori.bert.data_generate import parse_instruction
from order_matter_ori.bert.vocab import WordVocab
from order_matter_ori.config import USE_CUDA,CUDA_DEVICE
from util.common_helper import get_group_index



def load_block_encoder(                                            
        # model_path="model_file/block_embedding/bert_trained.model.ep4",
        model_path= "model_file/order_matter/bert/bert_train_complete.model.ep5",
        vocab_path="data/order_matter/vocab"
    ):
    encoder = Block_Encoder(model_path,vocab_path)
    return encoder


class Block_Encoder:
    def __init__(self, model_path, vocab_path):
        logging.info("Loading Vocab from :{}".format(vocab_path))
        self.vocab = WordVocab.load_vocab(vocab_path)
        logging.info("Vocab Size:{} ".format(len(self.vocab)))
        self.model = torch.load(model_path)
        logging.info("Load block encoder model from :{}".format(model_path))
        if USE_CUDA:
            self.model.cuda(CUDA_DEVICE)
        logging.info("Load block encoder model completed!")


    def encode(self, texts,seq_len=128):
        """
            text: block中各条指令用空格拼接组成的字符串
            seq_len: block 指令组成的字符串最大长度，大于进行截断，小于则用0进行填充
        """
        segment_label_l = []
        sequence_l = []
        
        for text in texts:
            sequence = []
            segment_label = []
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
            segment_label_l.append(segment_label)
            sequence_l.append(sequence)
        
         
        segment_label_l = torch.LongTensor(segment_label_l)
        sequence_l = torch.LongTensor(sequence_l)


        if USE_CUDA:
            sequence_l = sequence_l.cuda(CUDA_DEVICE)
            segment_label_l = segment_label_l.cuda(CUDA_DEVICE)
        
        encoded = self.model.forward(sequence_l, segment_label_l)
        # TODO:::::将每一个token的向量表示取均值作为整个block的向量表示
        result = torch.mean(encoded.detach(), dim=1)

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

                
def transform_block_embedding_graph(cfg_data_dict,block_embedding_model,save_path="data/order_matter/trainset_block_embedding_graph/"):
    """
        将原始cfg生成block embedding后的cfg图
        Parameters:
            origin_graph: 原始cfg图

    """
    G = nx.DiGraph()
    map_dict = {}
    block_ins_string_l = []
    # 将每个block中的每条指令创建一个节点，block内指令顺序建立边连接
    for index,node in enumerate(cfg_data_dict["node"]):
        block_address = node["addr"]
        map_dict[block_address] = index
        ins_l = [parse_instruction(ins['text']) for ins in node["text"]]
        block_ins_string = " ".join(ins_l)
        block_ins_string_l.append(block_ins_string)

    block_embeddings = []
    for i in range(0,len(block_ins_string_l),300):
        block_embeddings.extend(list(block_embedding_model.encode(block_ins_string_l[i:i+300])))  # 这里需要确定的是能不能进行完整的编码

    for index,block_embedding in enumerate(block_embeddings):
        G.add_node(index,embedding=block_embedding)
    # 复原边
    for edge in cfg_data_dict["edge"]:
        G.add_edge(map_dict[edge["from"]],map_dict[edge["to"]])

    if not os.path.exists(save_path):
            logging.info("创建目录：{}".format(save_path))
            os.makedirs(save_path)
    block_embedding_graph_file = os.path.join(save_path,str(cfg_data_dict['fid']))
    write_pickle(G,block_embedding_graph_file)
    return G

def generate_block_embedding_cfg_to_file(
        dataset_file,
        block_embedding_model_path="model_file/order_matter/bert/bert_4_task.model.ep5",
        vocab_path="data/order_matter/vocab",
        generate_file_path="data/order_matter/trainset_block_embedding_graph/",
        limit=1000000000,
    ):
    """
        遍历原始cfg文件目录，生成block embedding后的cfg图，存储到文件中供dataset直接读取
    """
        

    logging.info("【Generate Block Embeding Cfg to File】")
    save_samples = 0
    # 读取df_group
    df = pd.read_json(dataset_file,orient='reocrd',lines=True,nrows=limit)
    block_encoder = load_block_encoder(model_path=block_embedding_model_path,vocab_path=vocab_path)
    df = df[df.apply(lambda x:len(x.to_dict()["node"])>=5,axis=1)]
    df.progress_apply(lambda x:transform_block_embedding_graph(x.to_dict(),block_encoder,generate_file_path),axis=1)

    logging.info("Generate block embedding cfg nums:{}".format(df.shape[0]))          
    return save_samples


if __name__=="__main__":
    from util.log_helper import get_logger
    logger = get_logger("order_matter/block_embedding.log")
    generate_block_embedding_cfg_to_file(dataset_file="data/trainset_22w.json",
        block_embedding_model_path="model_file/order_matter/bert/bert_4_task.model.ep5",
        vocab_path="data/order_matter/vocab",
        generate_file_path="data/order_matter/train_json_block_embedding_graph/"
    )

    # generate_block_embedding_cfg_to_file("data/test_json/","data/order_matter/test_json_block_embedding_graph/")

