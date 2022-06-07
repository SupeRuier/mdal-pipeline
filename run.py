'''
Read the hyperparameters.json file which stores the experimental parameters.
Each line represents a model strategy pair and its corresponding parameters.
Execute the expariment line by line for each repetition index using the file 'main.py'.
'''
#%%
import os
import sys
import json
import subprocess

#%%

# darwin is for macos
ENV = "local" if sys.platform == "darwin" else "server"  

#%%
execute_file = 'main.py'
dataset_name = 'digits'  # dataset_name_list = ['double_inter_twin_moon','triple_inter_twin_moon','digits', 'amazon', 'imageCLEF', 'office_31', 'office_home', 'PACs']
model_list = ['SDL_separate','MDNet','MAN']    # ['DANN','SDL_joint','SDL_separate','MDNet','MAN', 'CAN'] ['DANN','SDL_joint', 'CAN'] ['SDL_separate','MDNet','MAN']
strategy_list = ['Random'] # ['Random', 'Uncertainty', 'BADGE', 'EGL', 'Coreset']
# repetition_index_range = range(0,3) # range(5,10) # for each pair, execute 0 to 19
repetition_index_range = [0]
device_str = "cuda:2"

#%%
# processing control
while(True):
    print()
    device_name = device_str
    print(f'We are going to submit the jobs: \nOn dataset: {dataset_name}. \nOn device: {device_name}. \nThe repitition indexes are: {repetition_index_range}.')
    print(f'The model list: {model_list}.')
    print(f'The strategy list: {strategy_list}.')
    
    in_content = input('Proceed?[y/n]：')
    if in_content == 'y':
        print('Processing...')
        break
    elif in_content == 'n':
        print('Exit!')
        exit(0)
    else:
        print('Please input exactly y/n !!!')

#%%
'''
The execution command contains the execute_file, the index of model-strategy pair, the repetition index range, parameter file address.
For example:
python3  main.py 3 11 hyperparameters/hyperparameters_caltech.json &
'''

# Make directory
folder_address = './result/'+ dataset_name + '/'
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

#%%
for strategy_name in strategy_list:
    for model_name in model_list:
        if ENV == 'local':
            for repetition_index in repetition_index_range:
                command = f'/Users/rui/miniconda3/envs/py3.8/bin/python {execute_file} --dataset_name={dataset_name} --model_name={model_name} --strategy_name={strategy_name} --repetition_index={repetition_index} --device={device_str}'
                subprocess.Popen(command, shell=True)
        else:
            # Server
            for repetition_index in repetition_index_range:
                command = f'/home/herui/miniconda3/envs/py3.8/bin/python {execute_file} --dataset_name={dataset_name} --model_name={model_name} --strategy_name={strategy_name} --repetition_index={repetition_index} --device={device_str}'
                subprocess.Popen(command, shell=True)

print('Already submit all the jobs!')