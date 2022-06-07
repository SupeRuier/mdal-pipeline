from abc import ABC, abstractmethod
import torch.nn as nn


#%%
class model_by_whole(nn.Module, ABC):
    '''
    This types of model should be one complete NN for all the domains.
    So the module should be an subclass of nn.Module.

    For two types of MDL models: ['DANN', 'SDL_joint']
    '''

    @abstractmethod
    def forward(self, x, d):
        """
        Neural network process forward.

        Parameters
        ----------
        x: the instance
        d: the corresponding domain
        """
