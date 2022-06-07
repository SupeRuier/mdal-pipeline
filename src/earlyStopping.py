'''
Adjusted from https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py.
'''

import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0.0001, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_score, model, use_loss = True):

        if use_loss == True:
            # when loss go up, early stop.
            score = val_score
    
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val_score, model)
            elif score > self.best_score - self.delta:
                self.counter += 1
                self.trace_func(f'-----EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(val_score, model)
                self.counter = 0

        else:
            # when performance go down, early stop.

            score = val_score
    
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val_score, model)
            elif score < self.best_score + self.delta:
                self.counter += 1
                self.trace_func(f'-----EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(val_score, model)
                self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        # TODO if doesn't encounter any performance decreasing, there will not be an parameter in the path.
        # However, this case is unlikely to happen, so we just record it here. We will handle this when we met the problem.
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss