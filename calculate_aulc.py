# %%
import seaborn as sns
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib
import itertools
import numpy as np
import warnings
warnings.filterwarnings('ignore')
matplotlib.use('Agg')
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300

sns.set(style="ticks")
# sns.set(style="whitegrid")

# %%
'''
Here in this file, we plan to plot the final version of learning curves.
1. a conbined figure of all the passive models. 6 sub-figure in total.
2. all the dataset-model in a single graph. (4*5)
3. The Results of the Best Stratrgy on Each Model 4 subfigure
4. On each dataset, the domain performances. 6 set of figs
'''

# %%


class Analyser:
    '''
    A class to process PerformanceLog instances.
    Plot and save the results.
    '''

    def __init__(self, result_folder_address, model_strategy_pair_list, repetition_index_range):
        self.result_folder_address = result_folder_address
        self.repetition_index_range = repetition_index_range
        self.model_strategy_pair_list = model_strategy_pair_list
        self.domain_names = None

        # The .result file is in the order dataset=>model_strategy_pair=>repetition=>domain=>performance
        # Re-build in the order.
        # dataset=>domain=>model_strategy_pair=>list_of_performance(merge repetitions)
        self.aulc_mean_dict, self.aulc_std_dict= self._load_results()
        # Put list_of_performance(for each repetition) into a dataframe, and remain the same dict structure.

    def _load_results(self):
        '''
        Load results from the corresponding folder and re-build to the required order.
        # dataset=>domain=>model_strategy_pair=>list_of_performance(merge repetitions)
        '''
        aulc_mean_dict = {}
        aulc_std_dict = {}

        for model_strategy_pair in self.model_strategy_pair_list:
            aulc_mean_dict[model_strategy_pair] = {}
            aulc_std_dict[model_strategy_pair] = {}

            temp_aulc_list = []
            for i in range(*self.repetition_index_range):
                result_address = f'{self.result_folder_address}result_{model_strategy_pair}_{i}.result'
                temp_performance_log = pickle.load(open(result_address, "rb"))

                # All domain performance
                performance_list = temp_performance_log.performance_dict[-1] # as a list
                temp_aulc = self._calculate_aulc(performance_list)
                temp_aulc_list.append(temp_aulc)

            aulc_mean_dict[model_strategy_pair] = np.mean(temp_aulc_list)*100
            aulc_std_dict[model_strategy_pair] = np.std(temp_aulc_list)*100


        return aulc_mean_dict, aulc_std_dict

    def _calculate_aulc(self, performance_list):

        temp_list = []
        temp1 = 0

        for i in performance_list:
            temp2 = i
            if temp1 != 0:
                temp_list.append((temp1+temp2)/2)
            temp1 = temp2
        
        aulc = np.mean(temp_list)

        return aulc

    def export_aulc(self, save_address=None):
        '''
        Plot learning curves for all domains in one.
        '''

        data = {'Model-Strategy-Pair':[], 'mean':[], 'std':[]}
        
        for pair in self.model_strategy_pair_list:
            data['Model-Strategy-Pair'].append(pair)
            data['mean'].append(self.aulc_mean_dict[pair])
            data['std'].append(self.aulc_std_dict[pair])
        
        df = pd.DataFrame.from_dict(data)

        writer = pd.ExcelWriter(os.path.join(os.getcwd(), f'{save_address}aulc.xlsx'))
        df = df.round(2)
        df.to_excel(writer)
        writer.save()


#%%

#%%

def plot_curves(root_folder, dataset, model_list, strategy_list, repetition_index_range):

    result_folder_address = f'{root_folder}/folder_result/'
    image_folder_address = f'{root_folder}/folder_image/'
    if not os.path.isdir(image_folder_address):
        os.mkdir(image_folder_address)

    ####################################
    #####################################
    # For each net on the current dataset

    save_address = f'{root_folder}/folder_image/2.active_parformances/'
    if not os.path.isdir(save_address):
        os.mkdir(save_address)

    model_strategy_pair_list = []

    for model in model_list:
        for strategy in strategy_list:
            # Need to check if the combination is valid.
            # Several models cannot adopted to many strategies.
            model_strategy_pair_list.append(f'{model}_{strategy}')
    
    analyser = Analyser(result_folder_address, model_strategy_pair_list, repetition_index_range)
    analyser.export_aulc(save_address = save_address)

    ######################################
    ######################################

    save_address = f'{root_folder}/folder_image/4.domain_performance/'
    if not os.path.isdir(save_address):
        os.mkdir(save_address)

    model_strategy_pair_list_dict = {'double_inter_twin_moon': ['DANN_Random', 'SDL_joint_Random', 'SDL_separate_Random', 'MDNet_Random', 'MAN_Random'],
                                     'triple_inter_twin_moon': ['DANN_Random', 'SDL_joint_Random', 'SDL_separate_Random', 'MDNet_Random', 'MAN_Random'],
                                     'digits': ['DANN_Uncertainty', 'SDL_joint_Uncertainty', 'SDL_separate_Uncertainty', 'MDNet_Uncertainty', 'MAN_Uncertainty', 'CAN_Uncertainty'],
                                     'amazon': ['DANN_Uncertainty', 'SDL_joint_Uncertainty', 'SDL_separate_Uncertainty', 'MDNet_Uncertainty', 'MAN_Uncertainty', 'CAN_Uncertainty'],
                                     'office_31': ['DANN_Uncertainty', 'SDL_joint_Uncertainty', 'SDL_separate_Uncertainty', 'MDNet_Uncertainty', 'MAN_Uncertainty','CAN_Uncertainty'],
                                     'imageCLEF': ['DANN_Random', 'SDL_joint_Random', 'SDL_separate_Uncertainty', 'MDNet_Uncertainty', 'MAN_Uncertainty'],
                                     'office_home': ['DANN_Uncertainty', 'SDL_joint_Uncertainty', 'SDL_separate_Uncertainty', 'MDNet_Uncertainty', 'MAN_Uncertainty', 'CAN_Uncertainty'],
                                     'PACs': ['DANN_Uncertainty', 'SDL_joint_Uncertainty', 'SDL_separate_Uncertainty', 'MDNet_Uncertainty', 'MAN_Uncertainty', 'CAN_Uncertainty']}


    # model_strategy_pair_list = model_strategy_pair_list_dict[dataset]
    # analyser = Analyser(result_folder_address,
    #                     model_strategy_pair_list, repetition_index_range)
    # analyser.plot_pairs_seperately(save_address=save_address, domain_performance = True, y_label = "Average Accuracy on the Domain")



# %%
if __name__ == '__main__':
    '''
    Provide the name of the task (folder name, bacause there might be amazon-1/amazon-2 for different hyper-parameters).
    Provide the repitition number.

    In the corresponding folder, create new folder to save the figures.
    '''
    dataset = 'PACs' # ['digits', 'amazon', 'office_31', 'imageCLEF', 'office_home', 'PACs']
    task_result_folder_name = 'DONE-PACs' 
    root_folder = f'all-results/{task_result_folder_name}'
    repetition_index_range = (0, 3)

    model_list = ['DANN','SDL_joint','SDL_separate','MDNet','MAN', 'CAN'] # ['DANN','SDL_joint','SDL_separate','MDNet','MAN', 'CAN']
    strategy_list = ['Random', 'Uncertainty', 'BADGE', 'EGL', 'Coreset'] #['Random', 'Uncertainty', 'BADGE', 'EGL', 'Coreset']

    plot_curves(root_folder, dataset, model_list, strategy_list, repetition_index_range)
