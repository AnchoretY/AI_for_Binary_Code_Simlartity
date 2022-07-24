'''
Author: AnchoretY
Date: 2022-06-24 05:46:33
LastEditors: Yhk
LastEditTime: 2022-07-23 01:05:35
'''
import os
import sys

sys.path.append(os.getcwd())


import re
import glob
from tqdm import tqdm

import logging

from order_matter_ori.bert.train_bert import bert_train
from order_matter_ori.block_embedding.block_embedding import generate_block_embedding_cfg_to_file
from order_matter_ori.function_embedding.similarity_function_find import generate_function_embedding_to_pickle,func_embedding_model_test
from order_matter_ori.function_embedding.train_function_embedding import train_func_embedding_model
from util.log_helper import get_logger


if __name__=="__main__":
    # make_wordvocab()
    logger = get_logger("order_matter/train_end2end_all_language.log")

    bert_epoch = 5
    siamese_gnn_epoch = 5

    origin_cfg_train_path = "../../data/Function_similarity/all_language/train_func.json"
    origin_cfg_test_path = "../../data/Function_similarity/all_language/test_func.json"
    train_group_file = "../../data/Function_similarity/all_language/train_group.csv"
    test_group_file = "../../data/Function_similarity/all_language/test_group.csv"

    # Block Embedding生成配置
    vocab_path = "data/order_matter/all_language_22w/bert_corpus/vocab"
    bert_model_path = "model_file/order_matter/all_language_22w/bert/bert_4_task.model"   # bert模型输出文件集基本格式，最终输出模型文件还会在后面加上轮数与损失
    block_embedding_train_file_path = "data/order_matter/all_language_22w/block_embedding/train/bert_{}/".format(bert_epoch)
    block_embedding_test_file_path = "data/order_matter/all_language_22w/block_embedding/test/bert_{}/".format(bert_epoch)

    
    func_embedding_model_path = "model_file/order_matter/all_language_22w/function_similarity/bert_{}/".format(bert_epoch)
    trainset_func_embedding_file_path = "data/order_matter/all_language_22w/func_embedding/train/bert_{}/function_embedding_epoch{}.pickle".format(bert_epoch,siamese_gnn_epoch)
    testset_func_embedding_file_path = "data/order_matter/all_language_22w/func_embedding/test/bert_{}/function_embedding_epoch{}.pickle".format(bert_epoch,siamese_gnn_epoch)
    func_embedding_test_result = "data/order_matter/all_language_22w/test_result/bert_{}".format(bert_epoch)

    # 1. bert模型训练
    # bert_model_path = bert_train(
    #     train_cfg_dataset = "data/order_matter/all_language_22w/bert_corpus/cfg_train.txt",
    #     # test_cfg_dataset = "data/order_matter/cfg_test.txt",
    #     train_append_dataset = "data/order_matter/all_language_22w/bert_corpus/cfg_train_append.pkl",
    #     # test_append_dataset = "data/order_matter/cfg_test_append.pickle",
    #     vocab_path = vocab_path,
    #     bert_embedding_size = 128,
    #     bert_seq_len = 128,
    #     epoch=bert_epoch,
    #     batch_size=512,
    #     lr = 1e-4,
    #     dataset_max_sample_nums=3000000,
    #     bert_model_save_path=bert_model_path
    # )
    # #bert_model_path = "model_file/order_matter/all_language_22w/bert/bert_4_task.model.ep9"
    # # 2. 使用bert模型对cfg基本块进行编码并将编码后的cfg图存储到指定路径的同名文件中（默认选择bert最后一轮训练的模型进行block embeding）
    # generate_block_embedding_cfg_to_file(
    #     dataset_file=origin_cfg_train_path,
    #     block_embedding_model_path=bert_model_path,
    #     vocab_path=vocab_path,
    #     generate_file_path=block_embedding_train_file_path,
    # )

    # generate_block_embedding_cfg_to_file(
    #     dataset_file=origin_cfg_test_path,
    #     block_embedding_model_path=bert_model_path,
    #     vocab_path=vocab_path,
    #     generate_file_path=block_embedding_test_file_path,
    # )

    # 4. 训练函数编码gnn模型
    func_embedding_model_path = train_func_embedding_model(
        dataset_path=block_embedding_train_file_path,
        group_file=train_group_file,
        func_embedding_model_save_path=func_embedding_model_path,
        node_feature_size=128,
        graph_feature_size=256,
        batch_size = 10,
        epoch = siamese_gnn_epoch
    )

    

    # func_embedding_model_path = "model_file/order_matter/all_language_22w/function_similarity/bert_5/siamese_network_gnn_trained_loss_0.1417.model.ep2"
    # 5. 生成函数使用模型向量化的结果
    func_embedding_train_file = generate_function_embedding_to_pickle(
        datapath=block_embedding_train_file_path,
        embedding_model_file_path=func_embedding_model_path,
        save_file=trainset_func_embedding_file_path,
    )

    func_embedding_test_file = generate_function_embedding_to_pickle(
        datapath=block_embedding_test_file_path,
        embedding_model_file_path=func_embedding_model_path,
        save_file=testset_func_embedding_file_path,
    )

    # func_embedding_train_file = trainset_func_embedding_file_path
    # func_embedding_test_file = testset_func_embedding_file_path
    # 5. 测试函数编码模型效果
    func_embedding_model_test(
        test_file_path= func_embedding_train_file,
        result_save_path = func_embedding_test_result,
        is_trainset=True,
        group_file=train_group_file
    )
    
    func_embedding_model_test(
        test_file_path= func_embedding_test_file,
        result_save_path = func_embedding_test_result,
        is_trainset=False,
        group_file = test_group_file
    )









