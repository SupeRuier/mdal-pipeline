from .strategy import Strategy
from parameter_handler import param_handler
import numpy as np
import time
from copy import deepcopy
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from src.logger import Logger
logger = Logger(logger = __name__)


#%%
class EGL(Strategy):
    def __init__(self):
        self.strategy_name = 'EGL'

    def select(self, current_state, **kwargs):
        '''
        Select instance basing on the logistic regression models.
        Sort the confidence value of all the instances from all the domain then select the most unconfident ones.
        '''
        model = kwargs['model']
        trainset = kwargs['trainset']

        full_domain_idx_length_list = []
        domain_names = current_state.domain_names
        for domain in domain_names:
            temp_index = current_state.unlabeled_info[domain]
            if len(temp_index) == 0:
                continue
            temp_domain = [domain] * len(temp_index)
            X_domain_unlabeled = current_state.build_unlabeled_matrix(trainset, domain) # a matrix of unlabeled data in the current domain in the order of temp index.
            grad_length = self.get_expect_grad_length_manual(X_domain_unlabeled, domain, model)
            full_domain_idx_length_list.extend(zip(temp_domain,temp_index,grad_length))
        
        full_domain_idx_length_list.sort(key=lambda tup: tup[2], reverse = True) # reverse = True, decending order.
        
        if len(full_domain_idx_length_list) > param_handler.AL_batch_size:
            selected_list = [(full_domain_idx_length_list[i][0], full_domain_idx_length_list[i][1]) for i in range(param_handler.AL_batch_size)]
        else:
            selected_list = [(tuple_item[0], tuple_item[1]) for tuple_item in full_domain_idx_length_list]

        return selected_list
 
    def get_expect_grad_length_manual(self, X_unlabeled, domain, model):
        
        device = param_handler.device
        num_class = param_handler.num_labels
        embd_dim = model.get_last_hidden_layer_size() # the last layer dim
        model.eval()
        num_instance = len(X_unlabeled)

        # get embedding
        gradient_len_all = []

        logger.info(f'Start calculating the grad length for each instance in domain: {domain}')
        start_time = time.perf_counter()

        if not torch.is_tensor(X_unlabeled):
            #  transform format
            X_unlabeled = torch.from_numpy(X_unlabeled)
        else:
            X_unlabeled = X_unlabeled.contiguous()

        X_unlabeled = TensorDataset(X_unlabeled)
        X_unlabeled_loader = DataLoader(dataset = X_unlabeled, batch_size = 128, shuffle = False, num_workers = 0)
        
        with torch.no_grad():

            for unlabeled_batch in X_unlabeled_loader:
                unlabeled_batch = unlabeled_batch[0].to(device, dtype=torch.float)
                out, last_layer_embd = model(unlabeled_batch, domain) 
                # out = out.data.cpu().numpy()
                batchProbs = F.softmax(out, dim=1).data.cpu().numpy()
                last_layer_embd = last_layer_embd.data.cpu().numpy()
                batch_instance_num = unlabeled_batch.shape[0]
                
                label_gradient_length_list = []
                for pseudo_label in range(num_class):
                    gradient_embedding = np.zeros([batch_instance_num, embd_dim * num_class])
                    for j in range(batch_instance_num):
                        for c in range(num_class):
                            if c == pseudo_label:
                                gradient_embedding[j][embd_dim * c : embd_dim * (c+1)] = deepcopy(last_layer_embd[j]) * (1 - batchProbs[j][c])
                            else:
                                gradient_embedding[j][embd_dim * c : embd_dim * (c+1)] = deepcopy(last_layer_embd[j]) * (-1 * batchProbs[j][c])

                    # calculate the length for this psudo label.
                    label_gradient_length = np.linalg.norm(gradient_embedding, ord=2, axis=1, keepdims=True)
                    label_gradient_length_list.append(label_gradient_length)

                # take weighted average to get expected grad length.
                label_gradient_length_matrix = np.hstack(label_gradient_length_list)
                batch_weight_grad_length = np.sum(batchProbs * label_gradient_length_matrix, axis=1)
                batch_weight_grad_length = list(batch_weight_grad_length)
                gradient_len_all.extend(batch_weight_grad_length)

        end_time = time.perf_counter()
        duration = end_time - start_time
        logger.info(f'Finish calculation with time {duration}')

        return gradient_len_all