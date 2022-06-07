#%%
'''
This file contains all the AL strategies.

Input & Output:
It would take the dataset, model, budget, current labeling state and the corresponding hyperparameters as input.
The output would be a set of instances selected by the strategy.
'''

#%%
import numpy as np
import time
import torch

from parameter_handler import param_handler
from src.logger import Logger
logger = Logger(logger = __name__)

#%% 
class Strategy:
    def __init__(self):
        self.catogory = 'ALStrategy'

    def select(self, current_state, **kwargs):
        '''
        Select instance basing on the logistic regression models.
        Sort the confidence value of all the instances from all the domain then select the most unconfident ones.
        '''
        pass

