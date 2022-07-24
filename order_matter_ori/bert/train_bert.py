'''
Author: Yhk
Date: 2022-05-26 05:13:29
LastEditors: Yhk
LastEditTime: 2022-07-21 10:58:06
Description: 
'''
import os
import sys
sys.path.append(os.getcwd())

import logging
from order_matter_ori.bert.vocab import WordVocab
from order_matter_ori.bert.dataset import BERTDataset
from order_matter_ori.bert.model.bert import BERT
from order_matter_ori.bert.trainer import BERTTrainer
from order_matter_ori.config import BERT_EMBEDDING_SIZE,BERT_SEQUENCE_LEN,BATCH_SIZE,LEARNING_RATE
from util.log_helper import get_logger


def make_wordvocab(
        train_cfg_dataset = "data/order_matter/cfg_train.txt",
        vocab_path = "data/order_matter/vocab",
        max_size=13000,
        min_freq=1
    ):
    logging.info("Make WordVocab....")
    with open(train_cfg_dataset, "r", encoding="utf-8") as f1:
        vocab = WordVocab(f1, max_size=max_size, min_freq=min_freq)

    logging.info("VOCAB SIZE:{}".format(len(vocab)))
    vocab.save_vocab(vocab_path)
    logging.info("WordVocab model save to: {}".format("vocab_path"))
    return vocab

def bert_train(
        train_cfg_dataset = "data/order_matter/cfg_train.txt",
        # test_cfg_dataset = "data/order_matter/cfg_test.txt",
        train_append_dataset = "data/order_matter/cfg_train_append.pickle",
        # test_append_dataset = "data/order_matter/cfg_test_append.pickle",
        vocab_path = "data/order_matter/vocab",
        bert_embedding_size = 128,
        bert_seq_len = 128,
        epoch=10,
        batch_size=128,
        lr=1e-4,
        dataset_max_sample_nums=10000000,
        bert_model_save_path="model_file/order_matter/bert/bert_train_complete.model",
    ):
    
    logging.info("【Block Embedding Model Train】")
    if not os.path.exists(vocab_path):
        vocab = make_wordvocab(
            train_cfg_dataset = train_cfg_dataset,
            vocab_path = vocab_path,
            max_size=13000,
            min_freq=1
        )
    else:
        logging.info("\tLoading Vocab from: {}".format(vocab_path))
        vocab = WordVocab.load_vocab(vocab_path)
        logging.info("\tVocab Size: {}".format(len(vocab)))


    logging.info("\tLoading Dataset")
    train_dataset = BERTDataset(train_cfg_dataset, train_append_dataset,vocab,seq_len=bert_seq_len,max_samples=dataset_max_sample_nums)
    # test_dataset  = BERTDataset(test_cfg_dataset,test_append_dataset,vocab ,seq_len=bert_seq_len,max_samples=dataset_max_sample_nums//2)
    logging.info("\tTrain sample_nums:{}".format(len(train_dataset)))
    # logging.info("\tTest sample_nums:{}".format(len(test_dataset)))
    
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=20)
    # test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=20)

    logging.info("\tBuilding BERT model")
    bert = BERT(len(vocab), hidden=bert_embedding_size, n_layers=12, attn_heads=8, dropout=0.0)

    logging.info("\tCreating BERT Trainer")
    trainer = BERTTrainer(bert, len(vocab), train_dataloader=train_data_loader,
                            lr=lr, betas=(0.9, 0.999), weight_decay=0.0, log_freq=1000)

    logging.info("\tTraining Start")
    for i in range(epoch):
        trainer.train(i)
        bert_model_path = trainer.save(i,file_path=bert_model_save_path)
    logging.info("Block Embedding Model Save to :{}".format(bert_model_save_path))
    logging.info("")
    return bert_model_path


from torch.utils.data import DataLoader
if __name__=="__main__":
    logger = get_logger("order_matter/train_bert.log")
    bert_epoch = 30
    train_cfg_dataset = "data/order_matter/cfg_train.txt"
    #test_cfg_dataset = "data/order_matter/cfg_test.txt"
    train_append_dataset = "data/order_matter/cfg_train_append.pkl"
    #test_append_dataset = "data/order_matter/cfg_test_append.pkl"
    vocab_path = "data/order_matter/vocab"
    bert_model_path = "model_file/order_matter/bert/bert_4_task.model"   # bert模型输出文件集基本格式，最终输出模型文件还会在后面加上轮数与损失

    bert_train(
        train_cfg_dataset = train_cfg_dataset,
        #test_cfg_dataset = test_cfg_dataset,
        train_append_dataset = train_append_dataset,
        #test_append_dataset = test_append_dataset,
        vocab_path = vocab_path,
        bert_embedding_size = 128,
        bert_seq_len = 128,
        epoch=bert_epoch,
        batch_size=128,
        lr = 1e-4,
        dataset_max_sample_nums=1000000,
        bert_model_save_path=bert_model_path
    )