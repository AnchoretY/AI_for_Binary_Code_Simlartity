'''
Author: AnchoretY
Date: 2022-06-24 05:46:33
LastEditors: AnchoretY
LastEditTime: 2022-07-20 07:35:02
'''
from email.policy import default
import os
import sys
import click
from sqlalchemy import outparam
sys.path.append(os.getcwd())

import pandas as pd
from order_matter_ori.bert.data_generate import generate_data,generate_data_append,generate_optimize_map
from order_matter_ori.bert.train_bert import bert_train
from order_matter_ori.block_embedding.block_embedding import generate_block_embedding_cfg_to_file
from order_matter_ori.function_embedding.train_function_embedding import train_func_embedding_model

@click.command()
@click.option('--trainset-file', 'trainset_file', help='training function data file', required=True)
@click.option("--trainset-group-file","trainset_group_file",help="",required=True)
@click.option("--bert-corpus-generate","bert_corpus_generate",help="wether generate bert train corpus",default=False)
@click.option('--vocab-path', 'vocab_path', help='vocab file path', default="data/order_matter/vocab")
@click.option("--bert-model-path","bert_model_path",help="bert model save path",default="model_file/order_matter/bert/bert_4_task.model")
@click.option("--bert-epochs","bert_epochs",help="bert train epoch",default=10)
@click.option("--block-embedding-train-file-path","block_embedding_train_file_path",help="generate block embedding train file path",default="data/order_matter/bert_epoch_10/train_json_block_embedding_graph_4_task/")
@click.option("--func-embedding-model-path","func_embedding_model_path",default="model_file/order_matter/function_similarity/origin/bert_epoch_10/")
@click.option("--func-embedding-model-epochs","func_embedding_model_epochs",help="function embedding model train epochs",default=5)
def cli(trainset_file,trainset_group_file,vocab_path,bert_trainset_generate,bert_model_path,bert_epochs,block_embedding_train_file_path,func_embedding_model_path,func_embedding_model_epochs):

    # TODO：optime_map融入，device选择和limit设置
    if bert_trainset_generate:
        df = pd.read_json(trainset_file,orient='records',lines=True)
        generate_data(df,output_file="data/order_matter/all_language/cfg_train.txt")
        generate_data_append(df,output_file="data/order_matter/all_language/cfg_train_append.pkl")
        generate_optimize_map(df,save_data="data/order_matter/optimize_map.pkl")

    # 1. bert模型训练
    bert_model_path = bert_train(
        train_cfg_dataset = "data/order_matter/cfg_train.txt",
        # test_cfg_dataset = "data/order_matter/cfg_test.txt",
        train_append_dataset = "data/order_matter/cfg_train_append.pkl",
        # test_append_dataset = "data/order_matter/cfg_test_append.pickle",
        vocab_path = vocab_path,
        bert_embedding_size = 128,
        bert_seq_len = 128,
        epoch=bert_epochs,
        batch_size=128,
        lr = 1e-4,
        dataset_max_sample_nums=1000000,
        bert_model_save_path=bert_model_path
    )
    # 2. 使用bert模型对cfg基本块进行编码并将编码后的cfg图存储到指定路径的同名文件中（默认选择bert最后一轮训练的模型进行block embeding）
    generate_block_embedding_cfg_to_file(
        dataset_file=trainset_file,
        block_embedding_model_path=bert_model_path,
        vocab_path=vocab_path,
        generate_file_path=block_embedding_train_file_path
    )

    # 4. 训练函数编码gnn模型
    func_embedding_model_path = train_func_embedding_model(
        dataset_path=block_embedding_train_file_path,
        group_file=trainset_group_file,
        func_embedding_model_save_path=func_embedding_model_path,
        node_feature_size=128,
        batch_size = 10,
        epoch = func_embedding_model_epochs
    )



if __name__=="__main__":
    bert_epoch = 10
    siamese_gnn_epoch = 5

    origin_cfg_train_path = "data/sample_combine/train_func.json"

    vocab_path = "data/order_matter/vocab"
    bert_model_path = "model_file/order_matter/bert/bert_4_task.model"   # bert模型输出文件集基本格式，最终输出模型文件还会在后面加上轮数与损失

    block_embedding_train_file_path = "data/order_matter/bert_epoch_10/train_json_block_embedding_graph_4_task/"

    train_group_file = "data/sample_combine/train_group.csv"
    func_embedding_model_path = "model_file/order_matter/function_similarity/origin/bert_epoch_10/"

    











