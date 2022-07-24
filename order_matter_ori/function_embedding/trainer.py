import os
import sys
sys.path.append(os.getcwd()) # 将当前文件路径加入项目目录

import tqdm
import torch
import logging
from numpy import mean
from torch import optim
from torch.utils.data import DataLoader
from order_matter_ori.function_embedding.dataset import FunctionSimliarityDataset,collate_fn
from order_matter_ori.function_embedding.SiameseNetwork import SiameseNetworkGnn,CosineContrastiveLoss
from order_matter_ori.mpnn import MPNN_Graph_Embedding


class SiameseNetworkGnnTrainer:

    def __init__(self, siamese_network_gnn: SiameseNetworkGnn, 
                 train_dataloader: DataLoader, test_dataloader: DataLoader = None,
                 lr: float = 1e-5, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=10000,log_freq: int = 100,device=0):
        """
            :param siamese_network_gnn: SiameseNetw orkGnn模型
            :param train_dataloader: train dataset data loader
            :param test_dataloader: test dataset data loader [can be None]

            :param lr: learning rate of optimizer
            :param betas: Adam optimizer betas
            :param weight_decay: Adam optimizer weight decay param
            :param with_cuda: traning with cuda
            :param log_freq: logging frequency of the batch iteration
        """
        # 判断是否存在GPU以及存在GPU数量,选择后续模型与数据要使用的设备
        cuda_condition = torch.cuda.is_available()
        self.device = torch.device("cuda:{}".format(device) if cuda_condition else "cpu")
        logging.info("\tUse device:{}".format(self.device))
        
       
        
        self.model = siamese_network_gnn.to(self.device)
        
        # 设置train、test dataloader
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        # 设置优化器和损失函数
        self.optim = optim.AdamW(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.criterion = CosineContrastiveLoss()
        
        # 设置日志生成频率
        self.log_freq = log_freq

        # 记录信息
        self.loss_epoch = -1              # 当前最后一轮的损失函数


        logging.info("\tSet log freq:{}".format(self.log_freq))
        logging.info("\tTotal Parameters:{}".format(sum([p.nelement() for p in self.model.parameters()])))
        logging.info("Dataloader init completed!")
        
    def train(self, epoch):
        return self.iteration(epoch, self.train_dataloader)

    def test(self, epoch):
        return self.iteration(epoch, self.test_dataloader, train=False)

    def iteration(self, epoch, data_loader, train=True):
        """
            遍历dataloader，进行训练或测试
                :param epoch: 现在的训练轮数索引
                :param data_loader: torch.utils.data.DataLoader
                :param train: 是否是训练
        """
        str_code = "train" if train else "test"

        # tqdm进度条格式设置
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")
        
        loss_l = []
        # 遍历数据进行训练或测试，记录效果
        for i, data in data_iter:
            # 数据放到指定device上
            graphs_input1,graphs_input2,labels = data[:3],data[3:6],data[6]
            graphs_input1 = [value.to(self.device) if torch.is_tensor(value) else value for value in graphs_input1]
            graphs_input2 = [value.to(self.device) if torch.is_tensor(value) else value for value in graphs_input2]
            
            graphs_input1[0] = graphs_input1[0].to(self.device)
            graphs_input2[0] = graphs_input2[0].to(self.device)
            labels = labels.to(self.device)

            ouput_embedding1,output_embedding2 = self.model(graphs_input1[:2],graphs_input2[:2])
            loss = self.criterion(ouput_embedding1,output_embedding2,labels)
            
            loss_l.append(loss.item())
            self.loss_epoch = mean(loss_l)

            # 训练模式下，进行反向传播
            if train:
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            post_fix = {
                "epoch": epoch,
                "iter": i,
                "loss": self.loss_epoch
            }
            # 记录loss变化
            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))
        return self.loss_epoch


    def save(self, epoch, file_path="model_file/order_matter/function_similarity/"):
        """
            储存当前的模型
        :param epoch: 现在的训练轮数索引
        :param file_path: 文件输出的路径，最终文件为：file_path+"ep%d" % epoch
        :return: 最终模型存储的文件路径
        """
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        output_path = file_path + "siamese_network_gnn_trained.model.ep%d" % epoch
        output_path = "{}siamese_network_gnn_trained_loss_{:.4}.model.ep{}".format(file_path,self.loss_epoch,str(epoch))
        torch.save(self.model.gnn.cpu(), output_path)
        self.model.gnn.to(self.device) 
        logging.info("EP:{} Model Saved on:{}".format(epoch, output_path))
        return output_path




