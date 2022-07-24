'''
Author: Yhk
Date: 2022-05-11 00:33:04
LastEditors: AnchoretY
LastEditTime: 2022-07-13 21:49:54
Description: 
'''
import os
import logging

def _filter_warnning(record):
    """
        过滤掉warnning级别的日志
    """
    if record.levelno==logging.WARNING:
        return False
    else:
        return True


def get_logger(log_filename,console_output_level="INFO",file_output_level="DEBUG"):
    """
        快速获得logger,INFO以上输出除warning外输出到控制台 
        Parameters:
        --------------
            log_file:log_file存储的文件名，后缀名为.log
        Retutn:
        --------------
            logger队对象，使用这个对象可以直接进行上面指定的日志管理
    """
    
    util_path,_ = os.path.split(os.path.realpath(__file__))
    
    log_path = os.path.join(os.path.dirname(util_path),"log")
    logger = logging.getLogger()
    logger.setLevel('DEBUG')        # 将debug级别以上的等级日志全都记录到logger
    
    BASIC_FORMAT = "【%(asctime)s】%(levelname)s: %(message)s"
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)

    chlr = logging.StreamHandler() # 输出到控制台的handler
    chlr.setFormatter(formatter)
    chlr.setLevel(console_output_level)  # 也可以不设置，不设置就默认用logger的level
    logger.addHandler(chlr)

    if log_filename:
        if log_filename.split(".")[-1]!="log":
            raise ValueError("log_filename must end with .log")
    
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        
        log_file = os.path.join(log_path,log_filename)
        
        fhlr = logging.FileHandler(log_file,encoding='utf-8',mode='a') # 输出到文件的handler
        fhlr.setFormatter(formatter)
        fhlr.setLevel(file_output_level)  # 也可以不设置，不设置就默认用logger的level
        logger.addHandler(fhlr)
    
    # 过滤告警级别的日志
    logger.addFilter(_filter_warnning)
    logging.info("Log File Save To:     {}".format(log_file))
    return logging.getLogger()