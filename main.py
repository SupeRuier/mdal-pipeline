#%%

'''
This script is to run experiments for (dataset, model, strategy, repetition) pairs.
The results are dumped at the corresponding log file.

This file only execute one line of the parameter files for one model_strategy pair in one repetition.
This file only execute one line of the parameter files for one model_strategy pair in one repetition.
This file only execute one line of the parameter files for one model_strategy pair in one repetition.
'''

#%%
import os
import numpy as np
import torch
import random
import pickle
import traceback

from parameter_handler import param_handler
from src.logger import Logger
from src.stateInfo import StateInfo
from src.datasetloader import readDatasets

from active.activeFramework_joint import main_loop as main_loop_SDL_joint
from active.activeFramework_dann import main_loop as main_loop_DANN
from active.activeFramework_separate import main_loop as main_loop_SDL_separate
from active.activeFramework_mdnet import main_loop as main_loop_MDNet
from active.activeFramework_man import main_loop as main_loop_MAN
from active.activeFramework_can import main_loop as main_loop_CAN

#%%
def seed_torch(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

#%%
def main():
    '''
    Took a dict parameter.
    Load the dataset.
    Execute the AL process.
    '''

    # set random seed to torch
    seed_torch(seed=param_handler.torch_seed)

    '''
    Prepare dataset in this slot.
    '''
    data = readDatasets(param_handler.dataset_name)

    # The variance of experiments are controled by initail labeled instance selection which list below.
    current_state = StateInfo(data, param_handler.dataset_name, initial_labeled_num = param_handler.initial_labeled_num, random_state = param_handler.repetition_index)
    logger.info('Load data done!!!')

    '''
    For the model-strategy pair execute AL process. 
    Save the result (performance_log instance) to the result folder.
    Different types of models are handeled in a different way.
    '''
    if param_handler.model_name == 'SDL_separate':
        performance_log = main_loop_SDL_separate(data, current_state)
    elif param_handler.model_name == 'MDNet':
        performance_log = main_loop_MDNet(data, current_state)
    elif param_handler.model_name == 'MAN':
        performance_log = main_loop_MAN(data, current_state)
    elif param_handler.model_name == 'DANN':
        performance_log = main_loop_DANN(data, current_state)
    elif param_handler.model_name == 'SDL_joint':
        performance_log = main_loop_SDL_joint(data, current_state)
    elif param_handler.model_name == 'CAN':
        performance_log = main_loop_CAN(data, current_state)
    else:
        raise Exception("Model isn't known!") 
        exit(1)

    pickle.dump(performance_log, open(param_handler.save_folder_address + 'folder_result/result_' + '_'.join([param_handler.model_name, param_handler.strategy_name, str(param_handler.repetition_index)]) +'.result', "wb"))

#%%

def make_directories():
    # Make directory
    folder_address = param_handler.save_folder_address
    folder_checkpoint_address = folder_address + 'folder_checkpoint/'
    folder_init_weight_address = folder_address + 'folder_init_weight/' 
    folder_result_address = folder_address + 'folder_result/'
    folder_log_address = folder_address + 'folder_log/'
    
    if not os.path.isdir(folder_address):
        os.mkdir(folder_address)
    if not os.path.isdir(folder_checkpoint_address):
        os.mkdir(folder_checkpoint_address)
    if not os.path.isdir(folder_init_weight_address):
        os.mkdir(folder_init_weight_address)
    if not os.path.isdir(folder_result_address):
        os.mkdir(folder_result_address)
    if not os.path.isdir(folder_log_address):
        os.mkdir(folder_log_address)

# %%
if __name__ == '__main__':
    
    '''
    This file only execute one line of the parameter files for one model_strategy pair in one repetition.
    This file only execute one line of the parameter files for one model_strategy pair in one repetition.
    This file only execute one line of the parameter files for one model_strategy pair in one repetition.
    '''

    # Make directories
    make_directories()

    # Set logger

    logger = Logger()
    log_file_address = param_handler.log_save_address
    logger.set_addr(log_file_address)
    logger.log_title()
    logger.param_handler_info(param_handler)

    '''
    Log the error message
    '''
    try:
        main()
    except Exception as e:
        logger.error(f"Main program error: {e}")
        logger.error(traceback.format_exc())


# %%

