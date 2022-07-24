'''
Author: Yhk
Date: 2022-05-05 03:19:28
LastEditors: Yhk
LastEditTime: 2022-05-12 02:42:54
Description: 
'''
import os
import sys
sys.path.append(os.getcwd()) # 将当前文件路径加入项目目录

import torch
import pymysql
import heapq
import glob
import pandas as pd
import numpy as np

import palmtree_embedding.eval_utils as ins_embedding_utils
from gmn.utils import get_batch_input,get_input
from block_embedding import transform_block_embedding_graph
from networkx.drawing.nx_agraph import read_dot



def get_function_embedding(cfg_file):
    palmtree = ins_embedding_utils.UsableTransformer(
            model_path="palmtree_embedding/palmtree/transformer.ep19", 
            vocab_path="palmtree_embedding/palmtree/vocab"
        )
    graph = read_dot(cfg_file)
    graph = transform_block_embedding_graph(palmtree,graph)
    graph_input = get_input(graph)
    gnn = torch.load("model_file/function_similarity/siamese_network_gnn_trained.model.ep4")
    function_embedding = gnn(**graph_input)
    return function_embedding

def connect_mysql_db():
    db = pymysql.connect(host='10.162.99.122',
                     user='root',
                     password='123456')
    return db


def read_function_embedding_data_from_mysql():
    db = connect_mysql_db()
    cursor = db.cursor()
    sql = "select * from function_database.function_information"
    try:
        cursor.execute(sql)
        results = cursor.fetchall()
        func_data = []
        for row in results:
            _,func_name,program,library,word_size,complier,optimization,function_embedding = row
            function_embedding = [float(i) for i in function_embedding.split(",")]
            func_data.append([func_name,program,library,word_size,complier,optimization,function_embedding])
            df_func_data = pd.DataFrame(func_data,columns=["func_name","program","library","word_size","complier","optimization","func_embedding"])
    except:
        print("Read function embedding data fail!")
    db.close()
    return df_func_data

def generate_function_embedding_to_mysql(datapath):
    db = connect_mysql_db()
    cursor = db.cursor()

    save_data_l = []
    for program in os.listdir(datapath):
        program_path = os.path.join(datapath,program)
        for library in os.listdir(program_path):
            libraray_path = os.path.join(program_path,library)
            complie_type_l = os.listdir(libraray_path)
            for complie_type in complie_type_l:
                complie_path = os.path.join(libraray_path,complie_type)
                cfg_filename_l = [os.path.basename(cfg_file) for cfg_file in glob.glob(os.path.join(complie_path,"cfg*"))]
                for filename in cfg_filename_l:
                    file = os.path.join(complie_path,filename)
                    func_embedding = get_function_embedding(file).cpu().detach().numpy()[0].tolist()
                    func_embedding = ",".join([str(i) for i in func_embedding])
                    word_size,optimizer,complier = complie_type.split("_")
                    func_name = filename.split("_",3)[-1].split(".")[0]
                    save_data_l.append([func_name,program,library,complier,optimizer,word_size,func_embedding])
    try:
        sql = "insert into function_database.function_information(function_name,program,library,complier,optimization,word_size,function_embedding) values (%s,%s,%s,%s,%s,%s,%s)"
        cursor.executemany(sql,save_data_l)
        cursor.commit()
        cursor.close()
    except:
        print("Save to mysql fail! Please Check!")
        db.rollback()
    db.close()

def cosine_similarity(x,y):
    """
        计算两个向量之间的余弦相似度
        Args：
            x: array，向量1
            y: array，向量2
    """
    num = x.dot(y.T)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return num / denom

def get_top_k_sim_func(func_embedding,embedding_l,func_name_l,k=5):
    """
        获取指定函数最相似的k个函数的函数名与相似度。
        Args：
            func_embedding: 要进行查找的函数编码
            embedding_l: 数据库中的函数编码列表或Series
            func_name_l: 数据库中的函数名列表或Series，与embedding_l中相对应
            k：获取与目标函数相似度最高的k个函数
    """
    this_func_embedding = np.array(func_embedding)
    sim_score_l = []
    for target in embedding_l:
        target = np.array(target)
        sim_score = cosine_similarity(this_func_embedding,target)
        sim_score_l.append(sim_score)
    top_k_index = heapq.nlargest(k, range(len(sim_score_l)), sim_score_l.__getitem__)
    top_k_func = [func_name_l[i] for i in top_k_index]
    top_k_value = heapq.nlargest(k,sim_score_l)
    return top_k_func,top_k_value

def predict_top_k_correct(func_name,predict_top_k_func):
    """
        预测的top k是否正确，用于验证模型效果
        Args:
            func_name: 函数真正名字
            predict_top_k_func:预测得到的与函数最相似的top k个名字
    """
    if func_name in predict_top_k_func:
        return True
    return False
if __name__=="__main__":
    df_func_data = read_function_embedding_data_from_mysql()
    df_func_data[["top_k_func","top_k_sim_score"]] = df_func_data.apply(lambda x:get_top_k_sim_func(x.func_embedding,df_func_data["func_embedding"],df_func_data["func_name"],5),axis=1,result_type='expand')
    df_func_data["predict"] = df_func_data.apply(lambda x:predict_top_k_correct(x.func_name,x.top_k_func),axis=1)