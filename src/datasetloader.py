'''
This file is used to pre-process the required datasets from their raw file to our expected form.
The expected form of datasets are into two parts, X_dict and Y_dict.
In each dict, the key of the dictionary is the name of domain, the values are fratures in X_dict and class labels in Y_dict.
The same possition in both dicts would refer to the same instance.
'''
#%%

import sys
import pickle
import numpy as np 

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from parameter_handler import param_handler

from src.logger import Logger
logger = Logger(logger = __name__)

#%%

ENV = "local" if sys.platform == "darwin" else "server"  

#%%
class readDatasets: 
    '''
    ONLY USED TO READ DATASET.
    All preprocessed and saved.
    Just read file "./processedData/MNIST_USPS_0.7/MNIST_USPS_0.pkl"
    The preprocessed file is in the dictionary form.
    '''
    def __init__(self, dataset_name):
        '''
        '''
        self.dataset_name = dataset_name

        repetition_index = 0
        self.repetition_index = repetition_index
        self.file_address =  "./processedData/%s/%s.pkl" %  (dataset_name, dataset_name) 

        # If the file exists, load the train/test tuple from pkl file to self.train_set, self.test_set
        try:
            # Note the loaded data are in dictionary
            self.train_set, self.validation_set, self.test_set = pickle.load(open(self.file_address, "rb"))
        except:
            logger.info("Didn't load the dataset successfully!!")

    def get_joint_trainset(self, current_state):
        return TrainSetJoint(self.train_set, current_state)

    def get_separate_trainset(self, current_state):
        return TrainSetSeparate(self.train_set, current_state)

    def get_joint_testset(self, current_state, validation = False):
        if validation == False:
            return TestSetJoint(self.test_set, current_state)
        else:
            return TestSetJoint(self.validation_set, current_state)

    def get_separate_testset(self, current_state, validation = False):
        if validation == False:
            return TestSetSeparate(self.test_set, current_state)
        else:
            return TestSetSeparate(self.validation_set, current_state)



#%%
# Build a dataset for each domain.
class TrainSetJoint(Dataset):
    '''
    All the data should be gathered together, not as a dictionary as we used before but a list of tuples.
    Eg. [(x1,y1,d1,l1),(x2,y2,d2,l2),...,(xn,yn,dn,ln)]
    d is the corresponding domain. l is the current label state.
    '''
    def __init__(self, train_set, current_state):

        self.type = 'joint'
        self.X_dict = train_set['X_dict']
        self.Y_dict = train_set['Y_dict']

        # 4 lists for x,y,d,l.
        self.x_list, self.y_list, self.d_list, self.l_list = self.stack_data(current_state)

    def stack_data(self, current_state):
        '''
        Stack the data from the dictionary into uniform lists.
        This method will return four lists X,Y,D,L.

        For each domain, X is darray, Y/D/L are lists.
        '''
        X = []
        Y = []
        D = []
        L = []
        for domain in param_handler.domains:
            X.append(self.X_dict[domain])
            Y += list(self.Y_dict[domain])
            L += current_state.get_domain_labeled_info_boolean(domain)
            D += [current_state.get_domain_names_to_id()[domain]] * len(self.Y_dict[domain])

        X = torch.from_numpy(np.vstack(X))

        return X,Y,D,L

    def __getitem__(self, idx):
        x,y,d,l = self.x_list[idx], self.y_list[idx], self.d_list[idx], self.l_list[idx]
        return x,y,d,l

    def __len__(self):
        return len(self.y_list)

#%%
class TestSetJoint(Dataset):
    '''
    This class prepares the data for testing.
    All the data should be gathered together, not as[idx] a dictionary as we used before but a list of tuples.
    Eg. [(x1,y1,d1,l1),(x2,y2,d2,l2),...,(xn,yn,dn,ln)]
    d is the corresponding domain. l is the current label state.
    '''

    def __init__(self, test_set, current_state):

        self.type = 'joint'
        self.X_dict = test_set['X_dict']
        self.Y_dict = test_set['Y_dict']

        # 4 lists for x,y,d,l.
        self.x_list, self.y_list, self.d_list = self.stack_data(current_state)

    def stack_data(self, current_state):
        '''
        Stack the data from the dictionary into uniform lists.
        This method will return four lists X,Y,D,L.

        For each domain, X is darray, Y/D/L are lists.
        '''
        X = []
        Y = []
        D = []
        num_domain_dict = {}
        for domain in param_handler.domains:
            X.append(self.X_dict[domain])
            Y += list(self.Y_dict[domain])
            D += [current_state.get_domain_names_to_id()[domain]] * len(self.Y_dict[domain])
            num_domain_dict[domain] = len(self.Y_dict[domain])
        X = torch.from_numpy(np.vstack(X))  

        self.num_domain_dict = num_domain_dict
        return X,Y,D

    def __getitem__(self, idx):
        x,y,d = self.x_list[idx], self.y_list[idx], self.d_list[idx]
        return x,y,d

    def __len__(self):
        return len(self.y_list)

    def num_domain_dict(self):
        return self.num_domain_dict

#%%
# Build a dataset for each domain.
class TrainSetSeparate():
    '''
    This class is not a subclass of dataset.
    This one contains several datasets from different domains.
    Each domain contains its own dataset.
    
    The data still should be in a list of tuples.
    Eg. [(x1,y1,d1,l1),(x2,y2,d2,l2),...,(xn,yn,dn,ln)]
    d is the corresponding domain. l is the current label state.
    '''
    def __init__(self, train_set, current_state):

        self.type = 'Separate'
        self.all_domain_data = {} # dict of dataset
        for domain_name in param_handler.domains:
            self.all_domain_data[domain_name] = TrainSetSeparateItem(train_set, domain_name, current_state)

    def get_loader_dict(self, batch_size, shuffle, num_workers, drop_last):
        
        loader_dict = {} # dict of dataloader

        for domain_name, domain_dataset in self.all_domain_data.items():
            loader_dict[domain_name] = DataLoader(domain_dataset, batch_size=batch_size, shuffle=shuffle, num_workers = num_workers, drop_last=drop_last)

        return loader_dict

#%%
# Build a dataset for each domain.
class TrainSetSeparateItem(Dataset):
    '''
    The data still should be in a list of tuples.
    Eg. [(x1,y1,d1,l1),(x2,y2,d2,l2),...,(xn,yn,dn,ln)]
    d is the corresponding domain. l is the current label state.
    '''
    def __init__(self, train_set, domain_name, current_state):

        self.X_dict = train_set['X_dict']
        self.Y_dict = train_set['Y_dict']

        # 4 lists for x,y,d,l.
        self.x_list, self.y_list, self.d_list, self.l_list = self.stack_data(domain_name, current_state)

    def stack_data(self, domain_name, current_state):
        '''
        Stack the data from the dictionary into uniform lists.
        This method will return four lists X,Y,D,L.

        For each domain, X is darray, Y/D/L are lists.
        '''
        X = []
        Y = []
        D = []
        L = []

        X = self.X_dict[domain_name]
        Y = list(self.Y_dict[domain_name])
        L = current_state.get_domain_labeled_info_boolean(domain_name)
        D = [current_state.get_domain_names_to_id()[domain_name]] * len(self.Y_dict[domain_name])

        return X,Y,D,L

    def __getitem__(self, idx):
        x,y,d,l = self.x_list[idx], self.y_list[idx], self.d_list[idx], self.l_list[idx]
        return x,y,d,l

    def __len__(self):
        return len(self.y_list)

#%% 
# Build a dataset for each domain.
class TestSetSeparate():
    '''
    This class is not a subclass of dataset.
    This one contains several datasets from different domains.
    Each domain contains its own dataset.
    
    The data still should be in a list of tuples.
    Eg. [(x1,y1,d1,l1),(x2,y2,d2,l2),...,(xn,yn,dn,ln)]
    d is the corresponding domain. l is the current label state.
    '''
    def __init__(self, test_set, current_state):

        self.type = 'Separate'
        self.all_domain_data = {}
        for domain_name in param_handler.domains:
            self.all_domain_data[domain_name] = TestSetSeparateItem(test_set, domain_name, current_state)

    def get_loader_dict(self, batch_size, shuffle, num_workers, drop_last):
        
        loader_dict = {}

        for domain_name, domain_dataset in self.all_domain_data.items():
            loader_dict[domain_name] = DataLoader(domain_dataset, batch_size=batch_size, shuffle=shuffle, num_workers = num_workers, drop_last=drop_last)

        return loader_dict

#%% 
# Build a dataset for each domain.
class TestSetSeparateItem(Dataset):
    '''
    The data still should be in a list of tuples.
    Eg. [(x1,y1,d1,l1),(x2,y2,d2,l2),...,(xn,yn,dn,ln)]
    d is the corresponding domain. l is the current label state.
    '''
    def __init__(self, test_set, domain_name, current_state):

        self.X_dict = test_set['X_dict']
        self.Y_dict = test_set['Y_dict']

        # 4 lists for x,y,d,l.
        self.x_list, self.y_list, self.d_list = self.stack_data(domain_name, current_state)

    def stack_data(self, domain_name, current_state):
        '''
        Stack the data from the dictionary into uniform lists.
        This method will return four lists X,Y,D,L.

        For each domain, X is darray, Y/D/L are lists.
        '''
        X = []
        Y = []
        D = []

        X = self.X_dict[domain_name]
        Y = list(self.Y_dict[domain_name])
        D = [current_state.get_domain_names_to_id()[domain_name]] * len(self.Y_dict[domain_name])

        return X,Y,D

    def __getitem__(self, idx):
        x,y,d = self.x_list[idx], self.y_list[idx], self.d_list[idx]
        return x,y,d

    def __len__(self):
        return len(self.y_list)