'''
Author: Yhk
Date: 2022-05-05 03:19:28
LastEditors: Yhk
LastEditTime: 2022-07-22 23:15:20
Description: 
'''
from multiprocessing import get_logger
import os
import sys
sys.path.append(os.getcwd()) # 将当前文件路径加入项目目录

import logging
import torch
import hnswlib
import numpy as np
import dgl

import pandas as pd
import numpy as np

from tqdm import tqdm
from util.common_helper import read_pickle,write_pickle,get_file_group_map
from order_matter_ori.mpnn.utils import get_batch_input
from util.log_helper import get_logger


def load_embedding_model(func_model_path):
    gnn = torch.load(func_model_path)
    return gnn

def get_function_embedding_batch(cfg_files,func_embedding_model):
    """
        批量获取函数的embedding
    """
    graphs = []
    cols = ["func_name"]
    func_info = []
    node_nums_l = []
    
    for cfg_file in cfg_files:
        filename = os.path.basename(cfg_file)
        graph = read_pickle(cfg_file)
        graph = dgl.from_networkx(graph,node_attrs=['embedding'])
        graph.edata["embedding"] = torch.ones(graph.num_edges(),4)

        node_nums = graph.num_nodes()
        if node_nums>5:
            graphs.append(graph)
            func_info.append([filename])
            node_nums_l.append(node_nums)
    df_func_info = pd.DataFrame(func_info,columns=cols)
    if len(graphs)==0:
        return 
    
    func_info = np.array(func_info)
    graphs_input = get_batch_input(graphs)[:2]
    function_embeddings = func_embedding_model(*graphs_input).cpu().detach().numpy()
    df_func_info["func_embedding"] = pd.Series(function_embeddings.tolist())
    df_func_info["node_nums"] = pd.Series(node_nums_l)

    return df_func_info
    


def generate_function_embedding_to_pickle(
        datapath,embedding_model_file_path,save_file="data/order_matter/function_embedding_trainset.pickle"
    ):
    logging.info("【Generate Function Embedding data to pickle】")
    batch_size = 1024
    file_l = []
    logging.info("\tLoad function embedding model form {}".format(embedding_model_file_path))
    embeding_model = load_embedding_model(embedding_model_file_path)
    file_l = os.listdir(datapath)
     
    func_data_df = pd.DataFrame([])
    for i in tqdm(range(0,len(file_l),batch_size)):
        batch_files = [os.path.join(datapath,filename) for filename in file_l[i:i+batch_size]]
        func_data_batch_df = get_function_embedding_batch(batch_files,embeding_model)
        if func_data_df.shape[0]==0:
            func_data_df = func_data_batch_df
        else:
            func_data_df = pd.concat([func_data_df,func_data_batch_df],axis=0)
    func_data_df = func_data_df.reset_index(drop=True)
    if not os.path.exists(os.path.dirname(save_file)):
        os.makedirs(save_file)
        logging.info("Create Dir:{}".format(os.path.dirname(save_file)))
    logging.info("\tWrite function embedding to {}".format(save_file))
    write_pickle(func_data_df,save_file)
    return save_file



def top_k_vector_find_hnsw(query_embedding,embedding_library,k=5):
    dim = len(embedding_library[0])
    num_elements = len(embedding_library)
    
    ids = np.arange(num_elements)
    
    # Declaring index
    p = hnswlib.Index(space = 'cosine', dim = dim) # possible options are l2, cosine or ip

    # Initializing index - the maximum number of elements should be known beforehand
    p.init_index(max_elements = num_elements, ef_construction = 200, M = 16)
    
    # Element insertion (can be called several times):
    p.add_items(embedding_library, ids)
    
    # Controlling the recall by setting ef:
    p.set_ef(50) # ef should always be > k

    # Query dataset, k - number of closest elements (returns 2 numpy arrays)
    labels, distances = p.knn_query(query_embedding, k = k)
    return labels,distances
    
def get_top_k_sim_func(query_embedding,embedding_library,embedding_library_func_names,k=5):
    """
        
    """
    top_k_funcs = []
    top_k_funcs_indexs,sim_scores = top_k_vector_find_hnsw(query_embedding,embedding_library,k)
    for i,top_k_index in enumerate(top_k_funcs_indexs):
        top_k_funcs.append([embedding_library_func_names[j] for j in top_k_index])
    return top_k_funcs,sim_scores.tolist(),top_k_funcs_indexs.tolist()
#     return sim_scores,top_k_funcs_indexs



def func_embedding_model_test(
        test_file_path,
        result_save_path = "data/order_matter/",
        is_trainset = False,
        group_file = "data/trainset_group.csv"
    ):
    def predict_top_k_correct(funcname,predict_top_k_func):
        # group_to_file_map,file_to_group_map = get_file_group_map(group_file)
        group_id = file_to_group_map[int(funcname)]
        group_func_l = group_to_file_map[group_id][::]
        group_func_l.remove(int(funcname))
        for predict_func in predict_top_k_func:
            if int(predict_func) in group_func_l:
                return True
        return False
    logging.info("【Function Embedding Effect Test】")
    if is_trainset:
        logging.info("Trainset:")
    else:
        logging.info("Testset:")
        
    group_to_file_map,file_to_group_map = get_file_group_map(group_file)
    df_func_data = read_pickle(test_file_path)
    sample_nums = df_func_data.shape[0]

    
    # tmp
    file_l = list(file_to_group_map.keys())
    logging.info("过滤前数量:{}".format(df_func_data.shape[0]))
    df_func_data[["func_name"]] = df_func_data[["func_name"]].astype(int)
    df_func_data = df_func_data[df_func_data.func_name.isin(file_l)]
    logging.info("过滤后数量:{}".format(df_func_data.shape[0]))


    logging.info("\t测试输入数据：{}".format(test_file_path))
    logging.info("\t查找样本总数:{}".format(sample_nums))
    # logging.info("\t源程序包含的函数个数：{}".format(df_func_data.drop_duplicates("func_name").shape[0]))

    embedding_l = df_func_data["func_embedding"].values.tolist()
    func_name_l = df_func_data["func_name"].values.tolist()
    top_5_func,top_5_sim_scores,top_5_funcs_indexs = get_top_k_sim_func(embedding_l,embedding_l,func_name_l,6)
    top_1_func,top_1_sim_scores,top_1_funcs_indexs = get_top_k_sim_func(embedding_l,embedding_l,func_name_l,2)

    df_func_data["top_1_func"] = top_1_func
    df_func_data["top_1_cos_dis"] = top_1_sim_scores
    df_func_data["top_1_func_index"] = top_1_funcs_indexs
    df_func_data["top_5_func"] = top_5_func
    df_func_data["top_5_cos_dis"] = top_5_sim_scores
    df_func_data["top_5_func_index"] = top_5_funcs_indexs
    
    df_func_data["top_1_predict"] = df_func_data.apply(lambda x:predict_top_k_correct(x.func_name,x.top_1_func),axis=1)
    top_1_correct_sample_nums = df_func_data[df_func_data["top_1_predict"]==True].shape[0]
    top_1_acc = top_1_correct_sample_nums/sample_nums
    logging.info("\ttop 1准确率：{}".format(top_1_acc))
    df_func_data["top_5_predict"] = df_func_data.apply(lambda x:predict_top_k_correct(x.func_name,x.top_5_func),axis=1)
    top_5_correct_sample_nums = df_func_data[df_func_data["top_5_predict"]==True].shape[0]
    top_5_acc = top_5_correct_sample_nums/sample_nums
    logging.info("\ttop 5准确率：{}".format(top_5_acc))
    if result_save_path.endswith("/"):
        result_save_path = result_save_path[:-1]

    if is_trainset:
        save_path = "{}/fun_sim_test_{}_{}_{}.pkl".format(result_save_path,"trainset",top_1_acc,top_5_acc)
    else:
        save_path = "{}/fun_sim_test_{}_{}_{}.pkl".format(result_save_path,"testset",top_1_acc,top_5_acc)

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(save_path)
        logging.info("Create Dir:{}".format(os.path.dirname(save_path)))
    write_pickle(df_func_data,save_path)
    logging.info("\tResult save into:{}".format(save_path))
    return save_path
    




if __name__=="__main__":
    # logger = get_logger("order_matter/trainset_test.log")


    # block_embedding_test_file_path = "data/order_matter/bert_epoch_10/test_json_block_embedding_graph_4_task/"
    # func_embedding_model_path = "model_file/order_matter/function_similarity/origin/bert_epoch_10/siamese_network_gnn_trained_loss_0.1408.model.ep2"
    # embedding_model_test_file_path = "data/order_matter/bert_epoch_10/function_embedding_testset_epoch0.pickle"

    # func_embedding_test_result = "data/order_matter/bert_epoch_10/"
    # group_file = "data/sample_combine/test_group.csv"
    # # 0 epoch acc 可达0.81

    # # 生成函数embedding pickle
    # func_embedding_test_file = generate_function_embedding_to_pickle(
    #     datapath=block_embedding_test_file_path,
    #     embedding_model_file_path=func_embedding_model_path,
    #     save_file=embedding_model_test_file_path,
    # )
    
    # # 测试函数embedding进行函数相似性查找效果
    # # func_embedding_test_file = embedding_model_test_file_path
    # func_embedding_model_test(
    #     test_file_path= func_embedding_test_file,
    #     result_save_path = func_embedding_test_result,
    #     is_trainset=False,
    #     group_file=group_file
    # )
    print(os.path.dirname("data/order_matter/all_language_22w/func_embedding/train/bert_5/function_embedding_epoch5.pickle"))