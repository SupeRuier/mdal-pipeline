from .strategy import Strategy
from parameter_handler import param_handler
import numpy as np
import time
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from copy import deepcopy

from src.logger import Logger
logger = Logger(logger = __name__)

#%%
class BADGE(Strategy):
    def __init__(self):
        self.strategy_name = 'BADGE'

    def select(self, current_state, **kwargs):
        '''
        Select instance basing on the logistic regression models.
        Sort the confidence value of all the instances from all the domain then select the most unconfident ones.
        '''
        model = kwargs['model']
        trainset = kwargs['trainset']

        full_domain_idx_list = []
        full_embedding_list = []
        domain_names = current_state.domain_names
        for domain in domain_names:
            temp_index = current_state.unlabeled_info[domain]
            if len(temp_index) == 0:
                continue
            temp_domain = [domain] * len(temp_index)
            X_domain_unlabeled = current_state.build_unlabeled_matrix(trainset, domain) # a matrix of unlabeled data in the current domain in the order of temp index.
            grad_embedding = self.get_grad_embedding_manual(X_domain_unlabeled, domain, model)
            full_domain_idx_list.extend(zip(temp_domain,temp_index))
            full_embedding_list.extend(grad_embedding)
        
        # get the select idx regrad to the full_domain_idx_list
        # use kmean++
        logger.info('Start to use kmean++ on the embedding.')
        start_time = time.perf_counter()
        selected_list = self.select_from_embedding(full_domain_idx_list, full_embedding_list, param_handler.AL_batch_size)
        end_time = time.perf_counter()
        duration = end_time - start_time
        logger.info(f'Finish selection with time {duration}')

        return selected_list

    # gradient embedding (assumes cross-entropy loss)
    def get_grad_embedding_manual(self, X_unlabeled, domain, model):
        device = param_handler.device
        num_class = param_handler.num_labels
        embd_dim = model.get_last_hidden_layer_size() # the last layer dim
        model.eval()
        num_instance = len(X_unlabeled)

        if not torch.is_tensor(X_unlabeled):
            #  transform format
            X_unlabeled = torch.from_numpy(X_unlabeled)
        else:
            X_unlabeled = X_unlabeled.contiguous()

        gradient_embedding = np.zeros([num_instance, embd_dim * num_class])
        X_unlabeled = TensorDataset(X_unlabeled)
        X_unlabeled_loader = DataLoader(dataset = X_unlabeled, batch_size = 128, shuffle = False, num_workers = 0)
        
        logger.info(f'Start calculating the grad embedding for each instance in domain: {domain}')
        start_time = time.perf_counter()
        
        with torch.no_grad():
            current_item_id = 0

            for unlabeled_batch in X_unlabeled_loader:
                unlabeled_batch = unlabeled_batch[0].to(device, dtype=torch.float)
                out, last_layer_embd = model(unlabeled_batch, domain) 
                # out = out.data.cpu().numpy()
                batchProbs = F.softmax(out, dim=1).data.cpu().numpy()
                last_layer_embd = last_layer_embd.data.cpu().numpy()
                maxInds = np.argmax(batchProbs,1)
                batch_instance_num = unlabeled_batch.shape[0]
                
                for j in range(batch_instance_num):
                    for c in range(num_class):
                        if c == maxInds[j]:
                            gradient_embedding[current_item_id+j][embd_dim * c : embd_dim * (c+1)] = deepcopy(last_layer_embd[j]) * (1 - batchProbs[j][c])
                        else:
                            gradient_embedding[current_item_id+j][embd_dim * c : embd_dim * (c+1)] = deepcopy(last_layer_embd[j]) * (-1 * batchProbs[j][c])
                current_item_id += batch_instance_num
        
        end_time = time.perf_counter()
        duration = end_time - start_time
        logger.info(f'Finish calculation with time {duration}')

        return gradient_embedding

    def select_from_embedding(self, full_domain_idx_list, full_embedding_list, AL_batch_size):
        '''
        parameters:
        --------------------------
        full_domain_idx_list: list of domain-index pair tuples. e.g, ('domain1',13)
        full_embedding_list: list of embeddings
        
        return:
        --------------
        selected_domain_idx_list: list of selected domain-index pair tuples

        '''
        # Check if there is unlabeled instance. If not, just reutrn an empty list.
        if len(full_domain_idx_list)==0:
            return full_domain_idx_list

        full_embedding_list = np.vstack(full_embedding_list)
        if len(full_embedding_list) > AL_batch_size:
            idx_list = self.init_centers(full_embedding_list, AL_batch_size)
            selected_domain_idx_list = [full_domain_idx_list[i] for i in idx_list]
        else:
            selected_domain_idx_list = full_domain_idx_list

        return selected_domain_idx_list

    def init_centers(self, X, K):
        '''
        kmeans ++ initialization
        from the BADGE code https://github.com/JordanAsh/badge/blob/master/query_strategies/badge_sampling.py
        '''
        from sklearn.metrics import pairwise_distances
        from scipy import stats

        ind = np.argmax([np.linalg.norm(s, 2) for s in X])
        mu = [X[ind]]
        indsAll = [ind]
        centInds = [0.] * len(X)
        cent = 0
        # print('#Samps\tTotal Distance')
        
        while len(mu) < K:
            if len(mu) == 1:
                D2 = pairwise_distances(X, mu).ravel().astype(float)
            else:
                newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)
                for i in range(len(X)):
                    if D2[i] >  newD[i]:
                        centInds[i] = cent
                        D2[i] = newD[i]
            # print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
            
            # if sum(D2) == 0.0: pdb.set_trace()
            D2 = D2.ravel().astype(float)
            Ddist = (D2 ** 2)/ sum(D2 ** 2)
            customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
            ind = customDist.rvs(size=1)[0]
            mu.append(X[ind])
            indsAll.append(ind)
            cent += 1
        
        gram = np.matmul(X[indsAll], X[indsAll].T)
        val, _ = np.linalg.eig(gram)
        val = np.abs(val)
        
        return indsAll