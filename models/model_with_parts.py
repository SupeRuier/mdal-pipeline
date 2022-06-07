from abc import ABC, abstractmethod

#%%
class model_by_integral_parts(ABC):
    '''
    This types of model should consist many module part.
    For different domains, the different module parts are combined to make inference.
    So each part of the module should be an subclass of nn.Module, but the whole module should not be a subclass of nn.Module.

    For three types of MDL models: ['MDNet', 'SDL_separate', 'MAN']
    '''

    @abstractmethod
    def __call__(self, x, domain_name):
        '''
        Here we use this call function to make inference for the selected fomain by using the specific module components.
        Only be used in test and query selection, so normally should be switched to eval mode.
        '''

    @abstractmethod
    def eval(self):
        '''
        Switch all the module component to eval mode.
        '''

    @abstractmethod
    def state_dict(self):
        '''
        Save the current state dict.
        '''

    @abstractmethod
    def get_classifier(self, domain):
        '''
        Get the classifier module to utilize.
        For example, to extract the gradient embedding in the classifier.
        '''

    @abstractmethod
    def zero_grad(self):
        '''
        Zero grad all the submodules
        '''