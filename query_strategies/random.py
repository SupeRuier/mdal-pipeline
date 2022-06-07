from .strategy import Strategy
from parameter_handler import param_handler
import numpy as np

#%%
class RandomStrategy(Strategy):
    '''
    General random selection.
    '''
    def __init__(self):
        self.strategy_name = 'Random'
    
    def select(self, current_state, **kwargs):
        '''
        Actually We only need current_state and AL_batch_size in this methods.
        This strategy doesn't garantee the select propotion of different domains are same. The domain with more unlabeled data would have more chance to be selected.
        '''
        full_list = []
        domain_names = param_handler.domains
        for domain in domain_names:
            temp_index = current_state.unlabeled_info[domain]
            temp_domain = [domain] * len(temp_index)
            full_list.extend(zip(temp_domain,temp_index))
        
        selected_list = []
        if len(full_list) > param_handler.AL_batch_size:
            # The rest instances are enough for another batch selection.
            selected_rel_idx = np.random.choice(range(len(full_list)), param_handler.AL_batch_size, replace=False)
            for idx in selected_rel_idx:
                selected_list.append(full_list[idx])
        else:
            # Not enough for a batch, return all the rests.
            selected_list = full_list
        return selected_list