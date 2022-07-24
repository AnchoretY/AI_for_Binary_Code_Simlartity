
import os
import torch
import logging
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader

from ..model import BERTLM, BERT
from .optim_schedule import ScheduledOptim
import tqdm


class BERTTrainer:
    """
    BERTTrainer make the pretrained BERT model with two LM training method.
        1. Masked Language Model : 3.3.1 Task #1: Masked LM
        2. Next Sentence prediction : 3.3.2 Task #2: Next Sentence Prediction

    """

    def __init__(self, bert: BERT, vocab_size: int,
                 train_dataloader: DataLoader, test_dataloader: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=10000,
                 log_freq: int = 10):
        """
        :param bert: BERT model which you want to train
        :param vocab_size: total word vocab size
        :param train_dataloader: train dataset data loader
        :param test_dataloader: test dataset data loader [can be None]
        :param lr: learning rate of optimizer
        :param betas: Adam optimizer betas
        :param weight_decay: Adam optimizer weight decay param
        :param with_cuda: traning with cuda
        :param log_freq: logging frequency of the batch iteration
        """

        # Setup cuda device for BERT training, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")

        # This BERT model will be saved every epoch
        self.bert = bert
        # Initialize the BERT Language Model, with BERT model
        self.model = BERTLM(bert, vocab_size).to(self.device)
        # Distributed GPU training if CUDA can detect more than 1 GPU
        if torch.cuda.device_count() > 1:
            logging.info("Using {} GPUS for BERT".format(torch.cuda.device_count()))
            self.model = nn.DataParallel(self.model)

        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.test_data = test_dataloader

        # Setting the Adam optimizer with hyper-param
        # self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim = AdamW(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim_schedule = ScheduledOptim(self.optim, self.bert.hidden, n_warmup_steps=warmup_steps)

        # Using Negative Log Likelihood Loss function for predicting the masked_token
        self.masked_criterion = nn.NLLLoss(ignore_index=0)
        self.nsp_criterion = nn.NLLLoss()
        self.big_criterion = nn.NLLLoss()
        self.gc_criterion = nn.NLLLoss()
        self.log_freq = log_freq

        logging.info("Total Parameters:{}".format(sum([p.nelement() for p in self.model.parameters()])))

    def train(self, epoch):
        self.iteration(epoch, self.train_data)

    def test(self, epoch):
        self.iteration(epoch, self.test_data, train=False)

    def iteration(self, epoch, data_loader, train=True):
        str_code = "train" if train else "test"

        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")


        avg_loss = 0.0

        for i, data in data_iter:
            data = {key: value.to(self.device) for key, value in data.items()}

            # 1. forward the next_sentence_prediction and masked_lm model
            mask_lm_output,next_sent_output,next_sent_mask_output,block_in_graph_ouput,graph_classfication_ouput = self.model.forward(
                data["bert_nsp_input"],data["bert_mlm_input"], data["segment_label"],
                data["bert_big_input"],data["bert_big_segment_label"],data["bert_gc_input"],data["bert_gc_segment_label"]
                )

            # 2-1. MLM任务损失
            mask_loss = self.masked_criterion(mask_lm_output.transpose(1, 2), data["bert_mlm_label"])

            # 2-2. NSP任务损失
            nsp_loss = self.nsp_criterion(next_sent_output, data["is_next_label"])
            nsp_mask_loss = self.nsp_criterion(next_sent_mask_output, data["is_next_label"])

            # 2-3 BIG任务损失
            big_loss = self.big_criterion(block_in_graph_ouput,data["bert_big_label"])
            # 2-4 GC任务损失
            gc_loss = self.gc_criterion(graph_classfication_ouput,data["bert_gc_label"])


            

            # 3 汇总损失
            loss = mask_loss+ nsp_loss + nsp_mask_loss+ big_loss + gc_loss
            avg_loss+=loss
            # 4. 反向传播
            if train:
                self.optim_schedule.zero_grad()
                loss.backward()
                self.optim_schedule.step_and_update_lr()

            post_fix = {
                "epoch": epoch,
                "iter": i,
                "NSP:": nsp_loss.item(),
                "NSP_MASK:": nsp_mask_loss.item(),
                "MLM:": mask_loss.item(),
                "BIG":big_loss.item(),
                "GC":gc_loss.item(),
                "LOSS":loss.item()
            }

            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))
        if train==True:
            logging.info("Train loss:{}\n".format(round(avg_loss.item()/(i+1),4)))
        else:
            logging.info("Test loss:{}\n".format(round(avg_loss.itemt()/(i+1),4)))


    def save(self, epoch, file_path="model_file/block_embedding/bert_trained.model"):
        output_path = file_path + ".ep%d" % epoch
        if not os.path.exists(output_path):
            logging.info("Create Path:{}".format(output_path))
            os.mkdir(output_path)
        torch.save(self.bert.cpu(), output_path)    # 存储只存储bert语言模型部分，三个子任务部分的网络并没有进行存储
        self.bert.to(self.device)   
        logging.info("EP:{} Model Saved on:{}".format(epoch, output_path))
        return output_path


