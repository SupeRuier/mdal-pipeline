from .strategy import Strategy
from parameter_handler import param_handler
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

from src.logger import Logger
logger = Logger(logger = __name__)

#%%
class Uncertainty(Strategy):
    def __init__(self):
        self.strategy_name = 'Uncertainty'

    def select(self, current_state, **kwargs):
        '''
        Select instance basing on the logistic regression models.
        Sort the confidence value of all the instances from all the domain then select the most unconfident ones.
        '''
        model = kwargs['model']
        trainset = kwargs['trainset']

        full_list = []
        domain_names = current_state.domain_names
        for domain in domain_names:
            temp_index = current_state.unlabeled_info[domain]
            if len(temp_index) == 0:
                continue
            temp_domain = [domain] * len(temp_index)
            X_domain_unlabeled = current_state.build_unlabeled_matrix(trainset, domain) # a matrix of unlabeled data in the current domain in the order of temp index.
            temp_score = self.score(X_domain_unlabeled, domain, model)

            full_list.extend(zip(temp_domain,temp_index,temp_score))
        full_list.sort(key=lambda tup: tup[2]) # from low to high

        if len(full_list) > param_handler.AL_batch_size:
            selected_list = [(full_list[i][0], full_list[i][1]) for i in range(param_handler.AL_batch_size)]
        else:
            selected_list = [(tuple_item[0], tuple_item[1]) for tuple_item in full_list]
        
        return selected_list

    def score(self, X_unlabeled, domain, model):
        '''
        This function transfer the decision matrix to required score for active selection.
        The lower the score, the more likely to be selected.
        For binary classification, instances near the hyperplane would have lower score (the lower the better).
        For multi-class classification, plan to use BvSB.

        Parameters
        ----------------
        confidence : The decision matrix of the corresponding SVM.
        X_unlabeled : tensor

        Return
        ---------------
        score : the score base on default criteria.
        '''

        '''
        Acquire the confidence function.
        '''
        model.eval()
        if not torch.is_tensor(X_unlabeled):
            #  transform format
            X_unlabeled = torch.from_numpy(X_unlabeled)
        else:
            X_unlabeled = X_unlabeled.contiguous()

        unlabeled_set = TensorDataset(X_unlabeled)
        unlabeled_set_loader = DataLoader(dataset = unlabeled_set, batch_size = param_handler.NN_batch_size, shuffle = False, num_workers = 0)

        # get confidence
        softmax_confidence_list = []
        with torch.no_grad():
            for unlabeled_batch in unlabeled_set_loader:
                unlabeled_batch = unlabeled_batch[0].to(param_handler.device, dtype=torch.float)

                try:
                    softmax_confidence_batch = model.get_softmax_batch(unlabeled_batch) 
                except:
                    softmax_confidence_batch = model.get_softmax_batch(unlabeled_batch, domain) 
                softmax_confidence_batch = softmax_confidence_batch.cpu().numpy()
                softmax_confidence_list.append(softmax_confidence_batch)

        confidence = np.vstack(softmax_confidence_list)
        # should be a tensor matrix with shape = (n, c), n is the number of unlabeled instance and c is the number of class
        # multiclass classification, need to estimate the probability.
        # The confidence is revealed as ovo probability.
        # The confidence has shape (n_samples, n_classes)
        score = np.sort(confidence, axis=1)
        # BvSB
        score = score[:,-1] - score[:,-2]
        return score
