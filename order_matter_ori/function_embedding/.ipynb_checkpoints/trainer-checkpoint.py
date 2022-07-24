import os
import sys
sys.path.append(os.getcwd()) # 将当前文件路径加入项目目录

import tqdm
import torch
import torch.nn as nn
from function_similarity.SiameseNetwork import SiameseNetworkGnn,CosineContrastiveLoss
from torch.utils.data import DataLoader
from torch import optim



class SiameseNetworkGnnTrainer:

    def __init__(self, siamese_network_gnn: SiameseNetworkGnn, 
                 train_dataloader: DataLoader, test_dataloader: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=10000,
                 log_freq: int = 10):
        """
        :param siamese_network_gnn: SiameseNetworkGnn模型
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
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")
        
        self.model = siamese_network_gnn
        if torch.cuda.device_count() > 1: # 存在多块GPU可用时，采用多块GPU进行训练
            print("Using %d GPUS for SiameseNetworkGnn" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)
        
        # 设置train、test dataloader
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        # 设置优化器和损失函数
        self.optim = optim.AdamW(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.criterion = CosineContrastiveLoss()

        # 设置日志生成频率
        self.log_freq = log_freq

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        self.iteration(epoch, self.train_dataloader)

    def test(self, epoch):
        self.iteration(epoch, self.test_dataloader, train=False)

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

        # 遍历数据进行训练或测试，记录效果
        for i, data in data_iter:
            graphs_input1,graphs_input2,labels = data

            graphs_input1 = {key: (value.to(self.device) if torch.is_tensor(value) else value) for key, value in graphs_input1.items()}
            graphs_input2 = {key: (value.to(self.device) if torch.is_tensor(value) else value) for key, value in graphs_input2.items()}
            labels = labels.to(self.device)
            # print(graphs_input1["node_features"])
            ouput_embedding1,output_embedding2 = self.model(graphs_input1,graphs_input2)
            loss = self.criterion(ouput_embedding1,output_embedding2,labels)

            # 训练模式下，进行反向传播
            if train:
                self.optim.zero_grad()
                loss.backward()

            post_fix = {
                "epoch": epoch,
                "iter": i,
                "loss:": loss.item(),
            }
            # 记录loss变化
            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))


    def save(self, epoch, file_path="model_file/function_similarity/"):
        """
            储存当前的模型
        :param epoch: 现在的训练轮数索引
        :param file_path: 文件输出的路径，最终文件为：file_path+"ep%d" % epoch
        :return: 最终模型存储的文件路径
        """
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        output_path = file_path + "siamese_network_gnn_trained.model.ep%d" % epoch
        torch.save(self.model.gnn.cpu(), output_path)
        self.model.gnn.to(self.device) 
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path



from function_similarity.dataset import FunctionSimliarityDataset,collate_fn
from function_similarity.SiameseNetwork import SiameseNetworkGnn

if __name__=="__main__":
    train_dataset = FunctionSimliarityDataset("data/train_data2/")
    train_dataloader = DataLoader(train_dataset,batch_size=2,collate_fn=collate_fn)
    siamese_network_gnn = SiameseNetworkGnn(128,4)
    trainer = SiameseNetworkGnnTrainer(siamese_network_gnn,train_dataloader=train_dataloader)
    epoch = 5
    for i in range(epoch):
        trainer.train(i)
        trainer.save(i)