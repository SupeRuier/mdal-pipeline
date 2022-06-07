import argparse

import json
import torch
import torch.nn as nn
import math

description = "Add parameters into the program."
parser = argparse.ArgumentParser(description=description)

##################################
##################################
##################################

# General Info

parser.add_argument('--dataset_name', type=str, help = 'dataset_name')
parser.add_argument('--model_name', type=str, help = 'model_name')
parser.add_argument('--strategy_name', type=str, help = 'strategy_name')
parser.add_argument('--debug_mode', type=bool, default=False)

# man_validation
parser.add_argument('--man_validation', type=bool, default=False)
parser.add_argument('--s_rate', type=float, default=1)
parser.add_argument('--p_rate', type=float, default=1)

# Dataset info

parser.add_argument('--domains', type=str, nargs='+', default=[])
parser.add_argument('--num_labels', type=int, default=2)

# Strategy info

parser.add_argument('--repetition_index', type=int, help = 'the index of the current repetition', default=666)
parser.add_argument('--AL_batch_size', type=int) # Here need to be changed
parser.add_argument('--total_budget', type=int) # Here need to be changed
parser.add_argument('--initial_labeled_num', type=int) # Here need to be changed
parser.add_argument('--train_from_begin', type=bool, default=True)
parser.add_argument('--lr_decay', type=bool, default=True)
parser.add_argument('--lr_update_rate', type=float, default=0.1)
parser.add_argument('--lr_update_patience', type=int, default=5)
parser.add_argument('--min_lr', type=float, default=1e-7)

# network training parameter

parser.add_argument('--device', type=str, default='cpu', help = 'which device the code would be processed on')
parser.add_argument('--torch_seed', type=int, default=1)
parser.add_argument('--dropout', type=float)
parser.add_argument('--weight_decay', type=float)
parser.add_argument('--epochs', type=int)
parser.add_argument('--NN_batch_size', type=int)
parser.add_argument('--learning_rate', type=float)
parser.add_argument('--D_learning_rate', type=float)
parser.add_argument('--patience', type=float)
parser.add_argument('--early_stop_by', type=str, default='loss') # could be loss or metric.
parser.add_argument('--optimizer', type=str, default='Adam') # Adam or SGD

# Model parameters
parser.add_argument('--dann_tradeoff', type=float, help='the trade off for DANN', default=0.1)
parser.add_argument('--man_tradeoff', type=float, help='the trade off for MAN', default=0.1)
parser.add_argument('--dacl_alpha', type=float, help='the trade off for DACL separation regularizer', default=0.1)
parser.add_argument('--dacl_gamma', type=float, help='the trade off for DACL refinement', default=0.1)
parser.add_argument('--can_lambda', type=float, help='the trade off for CAN', default=0.1)

# Performance Log

parser.add_argument('--delete_checkpoints', type=bool, default=True)
# parser.add_argument('--s_d_performance', type=bool, default=True)
# parser.add_argument('--s_d_decision_xlsx', type=bool, default=True)
# parser.add_argument('--train_s_d_decision_xlsx', type=bool, default=True)


##########################################
##########################################
##########################################
##########################################

param_handler = parser.parse_args()

# automatically prepared options
## folder address

save_folder_address = f'./result/{param_handler.dataset_name}/'
checkpoint_save_address = save_folder_address + 'folder_checkpoint/' + 'checkpoint_' + '_'.join([param_handler.model_name, param_handler.strategy_name, str(param_handler.repetition_index)]) + '.pt'
init_weight_save_address = save_folder_address + 'folder_init_weight/' + 'init_weight_' + '_'.join([param_handler.model_name, param_handler.strategy_name, str(param_handler.repetition_index)]) + '.pt'
log_save_address = save_folder_address + f'folder_log/{param_handler.model_name}_{param_handler.strategy_name}_{param_handler.repetition_index}.log'
result_save_address = save_folder_address + f'folder_result/{param_handler.model_name}_{param_handler.strategy_name}_{param_handler.repetition_index}.result'

param_handler.save_folder_address = save_folder_address
param_handler.checkpoint_address = checkpoint_save_address
param_handler.init_weight_address = init_weight_save_address
param_handler.log_save_address = log_save_address
param_handler.result_address = result_save_address

## select device
torch.device(param_handler.device if torch.cuda.is_available() else "cpu")

## record domains
if len(param_handler.domains) == 0:
    # use default domains
    if param_handler.dataset_name.lower() == 'digits':
        param_handler.domains = ['mnist', 'mnistm']
        param_handler.num_labels = 10
    else:
        param_handler.domains = None

## Read saved parameters.

#%%
'''
Load saved parameters from file.
'''
if param_handler.debug_mode == False:
    global_parameter_file_addr = './hyperparameters/global_param.json'
    with open(global_parameter_file_addr, 'r') as f:
        global_param = json.load(f)
    params = global_param[param_handler.dataset_name]

    param_handler.AL_batch_size = params['AL_batch_size'] 
    param_handler.total_budget = params['total_budget']
    param_handler.initial_labeled_num = params['initial_labeled_num']
    param_handler.epochs = params['epochs']
    param_handler.learning_rate = params['learning_rate']
    param_handler.weight_decay = params['weight_decay']
    param_handler.patience = params['patience']
    param_handler.NN_batch_size = params['NN_batch_size']
    param_handler.optimizer = params['optimizer']
    param_handler.lr_decay = params['lr_decay']
    if param_handler.lr_decay == True:
        param_handler.lr_update_patience = params['lr_update_patience']
        param_handler.lr_update_rate = params['lr_update_rate']