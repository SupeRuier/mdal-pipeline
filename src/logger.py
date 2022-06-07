'''
Personalize a log.
'''

import logging
import pickle
import os
import numpy as np
import time

class Logger(object):

    def __init__(self, logger = None, file_address = None):
        # 文件的命名  
        self.logger = logging.getLogger(logger)
        self.logger.setLevel(logging.INFO)
        # 日志输出格式
        # self.formatter = logging.Formatter('[%(asctime)s] - %(filename)s] - %(levelname)s: %(message)s')
        self.formatter = logging.Formatter('[%(asctime)s] - %(levelname)s: %(message)s')


        # 创建一个FileHandler，用于写到本地
        if file_address is not None:
            self.file_address = file_address
            fh = logging.FileHandler(self.file_address, 'a', encoding='utf-8') 
            fh.setLevel(logging.INFO)
            fh.setFormatter(self.formatter)
            self.logger.addHandler(fh)
            # 关闭打开的文件
            fh.close()

        # # 创建一个StreamHandler,用于输出到控制台
        # ch = logging.StreamHandler()
        # ch.setLevel(logging.INFO)
        # ch.setFormatter(self.formatter)
        # self.logger.addHandler(ch)

        # 这两行代码是为了避免日志输出重复问题
        # self.logger.removeHandler(ch)
        # self.logger.removeHandler(fh)

    def error(self, message):
        self.logger.error(message)

    def info(self, message):
        self.logger.info(message)

    def getlog(self):
        return self.logger
    
    def log_title(self):
        self.logger.info('################################################')
        self.logger.info('################################################')
        self.logger.info('Save experiment log here.')

    def param_handler_info(self, param_handler):
        for arg, value in sorted(vars(param_handler).items()):
            self.logger.info("Argument %s: %r", arg, value)

    def time_info(self, budget_assumption, train_time, test_time):
        epoch_log_time_info = f'-----{budget_assumption:<5} train: {train_time:.3e}, test: {test_time:.3e}'
        self.logger.info(epoch_log_time_info) 

    def epoch_loss_info(self, budget_assumption, epoch, clf_loss_train, clf_loss_validation, valid_performance_dict_item):
        epoch_log_info = f'-----{budget_assumption:<5} epoch:{epoch:<3}  train_loss:{clf_loss_train:.4f}  val_loss:{clf_loss_validation:.4f}  validation:{valid_performance_dict_item}'
        self.logger.info(epoch_log_info) 

    def budget_test_performance_info(self, budget_assumption, performance_dict_item):
        budget_log_info = f'CURRENT COST = {budget_assumption:<5}, PERFORMANCE_TEST = {performance_dict_item}'
        self.logger.info(budget_log_info) 

    def budget_train_performance_info(self, budget_assumption, domain_labeled_num_dict, domain_correct_labeled_num_dict):
        performance_train = ''
        total_correct = 0
        total_labeled = 0 
        for domain in list(domain_labeled_num_dict.keys()):
            performance_train += f'{domain}: {domain_correct_labeled_num_dict[domain]}/{domain_labeled_num_dict[domain]}, '
            total_correct += domain_correct_labeled_num_dict[domain]
            total_labeled += domain_labeled_num_dict[domain]

        performance_train += f'-1: {total_correct}/{total_labeled}'

        budget_train_log_info = f'CURRENT COST = {budget_assumption:<5}, PERFORMANCE_TRAIN = {"{"+performance_train+"}"}'
        
        self.logger.info(budget_train_log_info) 

    def budget_state_info(self, budget_assumption, current_state):
        budget_log_info = f'CURRENT COST = {budget_assumption:<5}, UPDATED_LABELED_NUM = {current_state.get_domain_labeled_num()}'
        self.logger.info(budget_log_info) 

    def set_addr(self, file_address):
        self.file_address = file_address
        fh = logging.FileHandler(self.file_address, 'a', encoding='utf-8')  # 这个是python3的
        fh.setLevel(logging.INFO)
        fh.setFormatter(self.formatter)
        self.logger.addHandler(fh)
        # 关闭打开的文件
        fh.close()

if __name__ == '__main__':
    # These codes are only used for test.
    cur_path = os.path.dirname(os.path.realpath(__file__))
    log_path = os.path.join(os.path.dirname(cur_path), 'test.logs')
    logger = Logger('__name__', log_path).getlog()
    logger.info('tttttttt')
    logger = Logger('__name__').getlog()
    logger.info('aaaa')