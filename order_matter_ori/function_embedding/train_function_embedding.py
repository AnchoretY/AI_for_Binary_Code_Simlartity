'''
Author: AnchoretY
Date: 2022-07-08 04:14:26
LastEditors: AnchoretY
LastEditTime: 2022-07-22 00:19:43
'''
import os
import sys
sys.path.append(os.getcwd())

import logging

from torch.utils.data import DataLoader
from util.plot_helper import plot_curve
from order_matter_ori.function_embedding.dataset import FunctionSimliarityDataset,collate_fn
from order_matter_ori.mpnn import MPNN_Graph_Embedding
from order_matter_ori.function_embedding.SiameseNetwork import SiameseNetworkGnn
from order_matter_ori.function_embedding.trainer import SiameseNetworkGnnTrainer


def train_func_embedding_model(
    dataset_path,
    group_file,
    func_embedding_model_save_path,
    node_feature_size=128,
    graph_feature_size = 128,
    batch_size = 256,
    lr = 0.0001,
    epoch = 20,
    device = 0,
    max_sample_nums=1000000
    ):
    logging.info("【Function embdding model train】")
    logging.info("\tLoad data from :{}".format(dataset_path))
    train_dataset = FunctionSimliarityDataset(dataset_path,group_file=group_file,max_sample_nums=max_sample_nums)
    logging.info("\tCreate dataloader...")
    train_dataloader = DataLoader(train_dataset,batch_size=batch_size,collate_fn=collate_fn,num_workers=10,pin_memory=True)

    mpnn_gnn = MPNN_Graph_Embedding(node_feature_size,graph_feature_size)

    logging.info("\tCreate SiameseNetworkGnn...")
    siamese_network_gnn = SiameseNetworkGnn(mpnn_gnn)
    logging.info("\tCreate SiameseNetworkGnnTrainer ...")
    trainer = SiameseNetworkGnnTrainer(siamese_network_gnn,train_dataloader=train_dataloader,device=device,lr=lr)
    logging.info("\tStart train...")
    loss_l = []
    for i in range(epoch):
        loss = trainer.train(i)
        loss_l.append(loss)
        model_file_path = trainer.save(i,func_embedding_model_save_path)
    
    plot_curve(range(epoch),loss_l,"loss","Loss","epoch","loss",save_name="func_embedding_model.png")
    logging.info("\tTrain Completed!")
    logging.info("Function Embedding Model Save to {}".format(func_embedding_model_save_path))
    logging.info("")
    return model_file_path





if __name__=="__main__":
    from util.log_helper import get_logger
    logger = get_logger("order_matter/train_func_similarity_model.log")

    group_file = "../../data/Function_similarity/all_language/train_group.csv"
    train_datapath = "data/order_matter/all_language_22w/block_embedding/train/bert_5"
    func_embedding_model_path = "model_file/order_matter/function_similarity/origin/"
    train_func_embedding_model(
        train_datapath,
        group_file,
        func_embedding_model_save_path=func_embedding_model_path,
        epoch=10,
        max_sample_nums=1000
    )


   