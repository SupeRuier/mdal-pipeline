'''
This file prepocess each dataset and dump the processed data in to 'processedData' folder as .pkl document
'''
#%%
import os
import pickle
import numpy as np
from numpy.lib.function_base import append 
import scipy.io as scio
import copy

import torch

#%% 

'''
This file need to be executed independently!!!!!
'''
processed_folder = '/Users/rui/Desktop/Code/MDAL_comparison/processedData/'

#%%
class PreprocessDatasets:
    '''
    ONLY USED TO PREPROCESS DATASETS AND SAVE TO THE SPECIFC FOLDERS.
    Each dataset need to be seperated to train and test set. 
    Each set have the following structure:
        {'X_dict':{'domain1':[...], 'domain2':[...]}, 'Y_dict':{'domain1':[...], 'domain2':[...]}}
    The dumped file should contains (train_set, test_set)
    '''
    def __init__(self):
        self.save_address = None

    def save_processed_files(self, name):
        '''
        Read data from the raw files.
        Pre-process the data.
        Save the file after extracted from the raw file.
        '''
        if name == 'digits':
            data = self.get_digits()
        else: 
            # Sythetic dataset
            data = None

        return data

    def get_dicts_tuple(self):
        '''
        Read from the saved file.
        Return X_dict and Y_dict.
        '''
        return self.data

    def get_digits(self):
        '''
        This dataset is pre-divided into train and test set.
        '''
        from torchvision.datasets import MNIST
        from mnistm import MNISTM
        import torchvision

        mnist_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])
        mnistm_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

        # Prepare dataset
        root = '/Users/rui/Desktop/WORKS/MDAL-Comparison/MDAL_comparison_new/rawData'
        temp_mnist_train = MNIST(root, train=True, transform=mnist_transform, target_transform=None, download=False)
        temp_mnistm_train = MNISTM(root, train=True, transform=mnistm_transform, target_transform=None, download=False)

        mnist_train, mnist_valid = torch.utils.data.random_split(temp_mnist_train, [50000, 10000], generator=torch.Generator().manual_seed(42))
        mnistm_train, mnistm_valid = torch.utils.data.random_split(temp_mnistm_train, [50000, 10000], generator=torch.Generator().manual_seed(42))

        mnist_test = MNIST(root, train=False, transform=mnist_transform, target_transform=None, download=False)
        mnistm_test = MNISTM(root, train=False, transform=mnistm_transform, target_transform=None, download=False)

        # Prepare tensor
        dataloader_mnist_train = torch.utils.data.DataLoader(dataset=mnist_train, batch_size=50000, shuffle=False)
        dataloader_mnistm_train = torch.utils.data.DataLoader(dataset=mnistm_train, batch_size=50000, shuffle=False)

        mnist_train_tensor_X, mnist_train_tensor_y = dataloader_mnist_train.__iter__().__next__()
        mnistm_train_tensor_X, mnistm_train_tensor_y = dataloader_mnistm_train.__iter__().__next__()

        dataloader_mnist_valid = torch.utils.data.DataLoader(dataset=mnist_valid, batch_size=10000, shuffle=False)
        dataloader_mnistm_valid = torch.utils.data.DataLoader(dataset=mnistm_valid, batch_size=10000, shuffle=False)

        mnist_valid_tensor_X, mnist_valid_tensor_y = dataloader_mnist_valid.__iter__().__next__()
        mnistm_valid_tensor_X, mnistm_valid_tensor_y = dataloader_mnistm_valid.__iter__().__next__()

        dataloader_mnist_test = torch.utils.data.DataLoader(dataset=mnist_test, batch_size=10000, shuffle=False)
        dataloader_mnistm_test = torch.utils.data.DataLoader(dataset=mnistm_test, batch_size=10000, shuffle=False)

        mnist_test_tensor_X, mnist_test_tensor_y = dataloader_mnist_test.__iter__().__next__()
        mnistm_test_tensor_X, mnistm_test_tensor_y = dataloader_mnistm_test.__iter__().__next__()

        '''
        Assign values into the dict.
        For MNIST data, need to reshape. 
        input_data = input_data.expand(input_data.data.shape[0], 3, 28, 28)
        '''
        
        # train set
        train_dict = {}
        train_dict['X_dict'] = {}
        train_dict['Y_dict'] = {}
        train_dict['X_dict']['mnist'] = mnist_train_tensor_X.expand(mnist_train_tensor_X.shape[0], 3, 28, 28)
        train_dict['X_dict']['mnistm'] = mnistm_train_tensor_X
        train_dict['Y_dict']['mnist'] = mnist_train_tensor_y
        train_dict['Y_dict']['mnistm'] = mnistm_train_tensor_y

        valid_dict = {}
        valid_dict['X_dict'] = {}
        valid_dict['Y_dict'] = {}
        valid_dict['X_dict']['mnist'] = mnist_valid_tensor_X.expand(mnist_valid_tensor_X.shape[0], 3, 28, 28)
        valid_dict['X_dict']['mnistm'] = mnistm_valid_tensor_X
        valid_dict['Y_dict']['mnist'] = mnist_valid_tensor_y
        valid_dict['Y_dict']['mnistm'] = mnistm_valid_tensor_y

        # test set
        test_dict = {}
        test_dict['X_dict'] = {}
        test_dict['Y_dict'] = {}
        test_dict['X_dict']['mnist'] = mnist_test_tensor_X.expand(mnist_test_tensor_X.shape[0], 3, 28, 28)
        test_dict['X_dict']['mnistm'] = mnistm_test_tensor_X
        test_dict['Y_dict']['mnist'] = mnist_test_tensor_y
        test_dict['Y_dict']['mnistm'] = mnistm_test_tensor_y

        data = (train_dict, valid_dict, test_dict)

        # Save to the file. 0 means the test set is pre-divided.
        folder_address = f"./processedData/digits/"
        if not os.path.isdir(folder_address):
            os.mkdir(folder_address)

        self.save_address = "./processedData/digits/digits.pkl" 
        pickle.dump(data, open(self.save_address, "wb"))

        return data


#%%
if __name__ == '__main__':

    preprocess_datasets = PreprocessDatasets()
    data = preprocess_datasets.save_processed_files('digits')
    data = preprocess_datasets.save_processed_files('PACs')
