from .strategy import Strategy
from parameter_handler import param_handler
import numpy as np
import time
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from copy import deepcopy
from sklearn.neighbors import NearestNeighbors
import pickle
from datetime import datetime

from src.logger import Logger
logger = Logger(logger = __name__)

#%%
class Coreset(Strategy):
    def __init__(self):
        self.strategy_name = 'Coreset'

    def select(self, current_state, **kwargs):
        '''
        Select instance basing on the logistic regression models.
        Sort the confidence value of all the instances from all the domain then select the most unconfident ones.
        '''
        model = kwargs['model']
        trainset = kwargs['trainset']

        full_domain_idx_list = []
        full_embedding_list = []
        full_labeled_boolean_list = []
        domain_names = current_state.domain_names
        for domain in domain_names:
            X_domain_all = current_state.build_all_matrix(trainset, domain) # a matrix of unlabeled data in the current domain in the order of temp index.
            num_instance = X_domain_all.shape[0]
            temp_domain = [domain] * num_instance
            temp_index = range(num_instance)
            labeled_boolean = current_state.get_domain_labeled_boolean(domain)

            embedding = self.get_embedding(X_domain_all, domain, model)
            full_domain_idx_list.extend(zip(temp_domain,temp_index))
            full_embedding_list.append(embedding)
            full_labeled_boolean_list.extend(labeled_boolean)
        
        full_embedding_matrix = np.vstack(full_embedding_list)

        # get the select idx regrad to the full_domain_idx_list
        # use coreset (k-center-greedy)
        logger.info('Start to apply coreset on the embedding.')
        start_time = time.perf_counter()
        selected_list = self.select_from_embedding(full_domain_idx_list, full_embedding_matrix, full_labeled_boolean_list, param_handler.AL_batch_size)
        end_time = time.perf_counter()
        duration = end_time - start_time
        logger.info(f'Finish selection with time {duration}')

        return selected_list

    # feature embedding
    def get_embedding(self, X, domain, model):

        # Should be the second last output in the classifier.
        # If the classifier is only one layer, it would be the output of the extractor.

        device = param_handler.device
        model.eval()

        if not torch.is_tensor(X):
            #  transform format
            X = torch.from_numpy(X)
        else:
            X = X.contiguous()

        X = TensorDataset(X)
        X_unlabeled_loader = DataLoader(dataset = X, batch_size = 128, shuffle = False, num_workers = 0)
        
        batch_embedding_list = []

        with torch.no_grad():
            for unlabeled_batch in X_unlabeled_loader:
                unlabeled_batch = unlabeled_batch[0].to(device, dtype=torch.float)
                _ , batch_embedding = model(unlabeled_batch, domain) 
                batch_embedding = batch_embedding.data.cpu().numpy()
                batch_embedding_list.append(batch_embedding)

        batch_embedding_matrix = np.vstack(batch_embedding_list)
        return batch_embedding_matrix

    def select_from_embedding(self, full_domain_idx_list, full_embedding_matrix, full_labeled_boolean_list, AL_batch_size):
        
        '''
        parameters:
        --------------------------
        full_domain_idx_list: list of domain-index pair tuples. e.g, ('domain1',13)
        full_embedding_matrix: matrix of embeddings
        full_labeled_boolean_list: whether the corresponding list is labeled.
        
        return:
        --------------
        selected_domain_idx_list: list of selected domain-index pair tuples

        '''
        # Check if there is unlabeled instance. If not, just reutrn an empty list.
        full_labeled_boolean_list = np.array(full_labeled_boolean_list).astype('bool')
        num_labeled = sum(full_labeled_boolean_list)
        num_all = len(full_labeled_boolean_list)

        if num_all - num_labeled <= AL_batch_size:
            # Don't need to select.
            # Just take the rest.
            select_idx_ = np.arange(num_all)[~full_labeled_boolean_list] # the real idx in the whole list
            selected_domain_idx_list = [full_domain_idx_list[i] for i in select_idx_]
        else:
            # Need select.
            idx_list = self.coreset_selection(full_embedding_matrix, full_labeled_boolean_list, AL_batch_size)
            selected_domain_idx_list = [full_domain_idx_list[i] for i in idx_list]

        return selected_domain_idx_list

    def coreset_selection(self, full_embedding_matrix, full_labeled_boolean_list, AL_batch_size):

        '''
        So we use the k-center greedy instead (subtitution for coreset).
        '''

        lb_flag = full_labeled_boolean_list.copy() # boolean list for current labeled instances.
        embedding = full_embedding_matrix
        total_instance_number = len(full_labeled_boolean_list)

        logger.info('calculate distance matrix')
        t_start = datetime.now()
        dist_mat = np.matmul(embedding, embedding.transpose())
        sq = np.array(dist_mat.diagonal()).reshape(total_instance_number, 1)
        dist_mat *= -2
        dist_mat += sq
        dist_mat += sq.transpose()
        dist_mat = np.sqrt(dist_mat)
        logger.info(f'Take time{datetime.now() - t_start}')

        logger.info('calculate greedy solution')
        t_start = datetime.now()

        mat = dist_mat[~lb_flag, :][:, lb_flag] # distances between unlabeled and labeled instances.

        verbose_num = 10
        if param_handler.dataset_name.lower() in ['pacs', 'digits', 'office_home']:
            verbose_num = 100

        for i in range(AL_batch_size):
            if i%verbose_num == 0:
                logger.info('greedy selection {}/{}'.format(i, AL_batch_size))
            mat_min = mat.min(axis=1)
            q_idx_ = mat_min.argmax() # the idx from the mat
            q_idx = np.arange(total_instance_number)[~lb_flag][q_idx_] # the real idx in the whole list
            lb_flag[q_idx] = True

            # re-construct the distances between unlabeled and labeled instances.
            mat = np.delete(mat, q_idx_, 0) 
            mat = np.append(mat, dist_mat[~lb_flag, q_idx][:, None], axis=1)

        logger.info(f'Take time{datetime.now() - t_start}')
        
        # Find the idx of the new selected instances by the xor between two boolean list.

        selected_index_list = np.arange(total_instance_number)[(full_labeled_boolean_list ^ lb_flag)]

        return selected_index_list

    def coreset_selection_old(self, full_embedding_matrix, labeled_boolean, AL_batch_size):

        '''
        This is the original version of the coreset selection.
        Is it very inconvenient to implement and apply.
        So we use the k-center greedy instead.
        '''

        lb_flag = labeled_boolean.copy() # boolean
        embedding = full_embedding_matrix.numpy()
        total_instance_number = embedding.shape[0]

        logger.info('calculate distance matrix')
        t_start = datetime.now()
        dist_mat = np.matmul(embedding, embedding.transpose())
        sq = np.array(dist_mat.diagonal()).reshape(len(self.X), 1)
        dist_mat *= -2
        dist_mat += sq
        dist_mat += sq.transpose()
        dist_mat = np.sqrt(dist_mat)
        logger.info(f'Take time{datetime.now() - t_start}')

        logger.info('calculate greedy solution')
        t_start = datetime.now()
        mat = dist_mat[~lb_flag, :][:, lb_flag]

        for i in range(AL_batch_size):
            if i%10 == 0:
                print('greedy solution {}/{}'.format(i, AL_batch_size))
            mat_min = mat.min(axis=1)
            q_idx_ = mat_min.argmax()
            q_idx = np.arange(total_instance_number)[~lb_flag][q_idx_]
            lb_flag[q_idx] = True
            mat = np.delete(mat, q_idx_, 0)
            mat = np.append(mat, dist_mat[~lb_flag, q_idx][:, None], axis=1)

        logger.info(f'Take time{datetime.now() - t_start}')
        opt = mat.min(axis=1).max()

        xx, yy = np.where(dist_mat <= opt)
        dd = dist_mat[xx, yy]

        lb_flag_ = self.idxs_lb.copy()
        subset = np.where(lb_flag_==True)[0].tolist()

        SEED = 5

        pickle.dump((xx.tolist(), yy.tolist(), dd.tolist(), subset, float(opt), AL_batch_size, total_instance_number), open('mip{}.pkl'.format(SEED), 'wb'), 2)


        # solving MIP
        # download Gurobi software from http://www.gurobi.com/
        # sh {GUROBI_HOME}/linux64/bin/gurobi.sh < core_set_sovle_solve.py

        sols = pickle.load(open('sols{}.pkl'.format(SEED), 'rb'))

        if sols is None:
            q_idxs = lb_flag
        else:
            lb_flag_[sols] = True
            q_idxs = lb_flag_
        print('sum q_idxs = {}'.format(q_idxs.sum()))

        return np.arange(self.n_pool)[(self.idxs_lb ^ q_idxs)]
