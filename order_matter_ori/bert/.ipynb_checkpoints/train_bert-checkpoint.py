'''
Author: Yhk
Date: 2022-05-26 05:13:29
LastEditors: AnchoretY
LastEditTime: 2022-06-02 05:30:29
Description: 
'''
import os
import sys
sys.path.append(os.getcwd())

import logging
from order_matter.bert.vocab import WordVocab
from order_matter.bert.dataset import BERTDataset
from order_matter.bert.model.bert import BERT
from order_matter.bert.trainer import BERTTrainer
from order_matter.config import BERT_EMBEDDING_SIZE,BERT_SEQUENCE_LEN,BATCH_SIZE,LEARNING_RATE
from util.log_helper import get_logger

logger = get_logger("order_matter/train_bert.log")

from torch.utils.data import DataLoader
if __name__=="__main__":
    

    train_cfg_dataset = "data/order_matter/cfg_train.txt"
    vocab_path = "data/order_matter/vocab"

    logging.info("Make WordVocab....")
    with open(train_cfg_dataset, "r", encoding="utf-8") as f1:
        vocab = WordVocab(f1, max_size=13000, min_freq=1)

    logging.info("VOCAB SIZE:{}".format(len(vocab)))
    vocab.save_vocab(vocab_path)
    logging.info("WordVocab model save to: {}".format("vocab_path"))

    logging.info("Loading Vocab from: {}".format(vocab_path))
    vocab = WordVocab.load_vocab(vocab_path)
    logging.info("Vocab Size: {}".format(len(vocab)))


    logging.info("Loading Train Dataset")
    train_dataset = BERTDataset(train_cfg_dataset, vocab,seq_len=BERT_SEQUENCE_LEN)
    logging.info("Sample_nums:{}".format(len(train_dataset)))
    
    logging.info("Creating Dataloader")
    train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=10)

    logging.info("Building BERT model")
    bert = BERT(len(vocab), hidden=BERT_EMBEDDING_SIZE, n_layers=12, attn_heads=8, dropout=0.0)

    logging.info("Creating BERT Trainer")
    trainer = BERTTrainer(bert, len(vocab), train_dataloader=train_data_loader,
                            lr=LEARNING_RATE, betas=(0.9, 0.999), weight_decay=0.0,
                            with_cuda=True, cuda_devices=[0], log_freq=100)


    logging.info("Training Start")
    for epoch in range(5):
        trainer.train(epoch)
        trainer.save(epoch)