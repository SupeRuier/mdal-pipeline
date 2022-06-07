#%%
import torch
from torch.utils.data import DataLoader
import numpy as np
import os

from src.logger import Logger
from parameter_handler import param_handler
from src.performanceMetric import PerformanceLog
from src.earlyStopping import EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau

from query_strategies.strategy_selector import strategy_selector
from models.model_selector import model_selector
from active.functions import labeled_set_ce_loss

logger = Logger(logger = __name__)

#%%
def main_loop(data, current_state):

    global device 
    device = param_handler.device
    NN_batch_size = param_handler.NN_batch_size
    AL_batch_size = param_handler.AL_batch_size
    patience_num = param_handler.patience
    budget_assumption = param_handler.initial_labeled_num 
    performance_log = PerformanceLog(current_state.get_domain_id_to_names())
    total_budget = param_handler.total_budget

    strategy_instance = strategy_selector(param_handler.strategy_name)

    '''
    Model / optimizer elements and the optimizer.
    '''
    model_dict = {}
    optim_dict = {}
    for domain_name in param_handler.domains:
        model_instance = model_selector('SDL_separate_individual', param_handler.dataset_name).to(device = device)
        model_dict[domain_name] = model_instance
        if param_handler.optimizer == 'Adam':
            optim_dict[domain_name] = torch.optim.Adam(model_instance.parameters(), lr = param_handler.learning_rate, weight_decay=param_handler.weight_decay)
        elif param_handler.optimizer == 'SGD':
            optim_dict[domain_name] = torch.optim.SGD(model_instance.parameters(), lr = param_handler.learning_rate, weight_decay=param_handler.weight_decay)
        else:
            raise Exception("Optimizer isn't known!") 


    model_instance = model_selector('SDL_separate', param_handler.dataset_name)
    model_instance.add_individual_dict(model_dict)

    torch.save(model_instance.state_dict(), param_handler.init_weight_address)

    '''
    Prepare test set. NOT a typortch dataset type.
    '''
    test_set = data.get_separate_testset(current_state)
    test_loader_dict = test_set.get_loader_dict(batch_size=NN_batch_size, shuffle=False, num_workers = 0, drop_last=False)

    validation_set = data.get_separate_testset(current_state, validation=True)
    validation_loader_dict = validation_set.get_loader_dict(batch_size=NN_batch_size, shuffle=False, num_workers = 0, drop_last=False)

    if total_budget == -1:
        total_budget = current_state.get_max_budget()

    '''
    The AL process, on the unlabeled instances of training set.
    '''
    while budget_assumption <= total_budget: 
        
        # At the beginning of each AL iteration.
        # initialize the the current model's parameters.
        if param_handler.train_from_begin == True:
            model_instance.load_state_dict(torch.load(param_handler.init_weight_address))

        # Build train loader in each iteration with the updated current state
        # The training istances would contains (x,y,d,l)
        train_set = data.get_separate_trainset(current_state)
        train_loader_dict = train_set.get_loader_dict(batch_size=NN_batch_size, shuffle=True, num_workers = 0, drop_last=False)

        # Train the NN
        early_stopping = EarlyStopping(patience=patience_num, verbose=False, path=param_handler.checkpoint_address, trace_func=logger.info)

        if param_handler.lr_decay == True:
            scheduler_list =[ReduceLROnPlateau(optim, mode='min', factor=param_handler.lr_update_rate, patience = param_handler.lr_update_patience, min_lr=param_handler.min_lr) for optim in optim_dict.values()]

        for epoch in range(param_handler.epochs):
            # train
            train(model_dict, train_loader_dict, optim_dict)
            clf_loss_train, _ , _ = train_performance(model_dict, train_loader_dict, current_state)
            # test on validation in epoch
            valid_performance_dict_item, clf_loss_validation = test(model_dict, validation_loader_dict, current_state)

            # log for epoch info
            logger.epoch_loss_info(budget_assumption, epoch, clf_loss_train, clf_loss_validation, valid_performance_dict_item)

            # Use the total accuracy or loss to check if to apply early stopping.
            if param_handler.early_stop_by == 'loss':
                # Use the total accuracy to check if to apply early stopping.
                early_stopping(clf_loss_validation, model_instance)
            else:
                # Use the current metric to check if to apply early stopping.
                early_stopping(valid_performance_dict_item[-1], model_instance, use_loss = False)

            if param_handler.lr_decay == True:
                for scheduler in scheduler_list:
                    scheduler.step(clf_loss_validation)

                if early_stopping.counter % (param_handler.lr_update_patience + 1) == 0 and early_stopping.counter != 0: # The next term after lr decay patience will decay.
                    logger.info("-----Update learning rate!!!")
                    model_instance.load_state_dict(torch.load(early_stopping.path))

            if early_stopping.early_stop:
                logger.info("-----EARLYSTOPPING!!!")
                break

        # load the corresponding parameters.
        model_instance.load_state_dict(torch.load(early_stopping.path))
        if param_handler.lr_decay == True:
            for optim in optim_dict.values():
                optim.param_groups[0]['lr']  = param_handler.learning_rate

        # final train set performance
        _, domain_labeled_num_dict, domain_correct_labeled_num_dict = train_performance(model_dict, train_loader_dict, current_state)
        logger.budget_train_performance_info(budget_assumption, domain_labeled_num_dict, domain_correct_labeled_num_dict)

        # test
        test_performance_dict_item, _ = test(model_dict, test_loader_dict, current_state)
        performance_log.add_performance_item(budget_assumption, test_performance_dict_item)

        # Log the iteration performance.  
        logger.budget_test_performance_info(budget_assumption, test_performance_dict_item)   
        # Log the current labeled instance number in each domain.
        logger.budget_state_info(budget_assumption, current_state)

        '''
        Query step
        '''
        if budget_assumption == total_budget:
            # run out of budget
            break
        else:
            # budget_assumption < total_budget
            budget_assumption += AL_batch_size
            if budget_assumption > total_budget:
                budget_assumption = total_budget

        # AL selection, in a dict form as normal.
        selected_list = strategy_instance.select(current_state, trainset = train_set, model = model_instance)

        # Update current state
        if len(selected_list) != 0:
            # if len(selected_list) == 0
            # The loop should stop. Not enough for a batch. Do nothing.
            current_state.update(selected_list)

    if param_handler.delete_checkpoints == True:
        os.remove(param_handler.init_weight_address)
        os.remove(param_handler.checkpoint_address)
    return performance_log


#%%

def train(net_dict, loader_dict, optim_dict):
    
    for domain_name in param_handler.domains:
        net = net_dict[domain_name]
        loader = loader_dict[domain_name]
        optim = optim_dict[domain_name]
        
        net.train()
    
        # Batch mode training, iteration.
        for _, (x,y,d,l) in enumerate(loader):
            x = x.to(device, dtype=torch.float)
            y,d,l = (i.to(device, dtype=torch.long) for i in [y,d,l])
            y_pred, _ = net(x)
            labeled_set_loss, labeled_num = labeled_set_ce_loss(y_pred, y, l)

            # Scalar 0, only for supervised model. Means unlabeled batch.
            # Just skip this batch
            if labeled_num == 0:
                continue

            optim.zero_grad()
            labeled_set_loss.backward()
            optim.step()

#%%
def train_performance(net_dict, loader_dict, current_state):
    
    batch_clf_loss = []
    batch_labeled_num = []

    domain_labeled_num_dict = {}
    domain_correct_labeled_num_dict = {}

    domain_int_list = list(current_state.get_domain_ids())
    for domain_int in domain_int_list:
        domain_labeled_num_dict[domain_int] = 0
        domain_correct_labeled_num_dict[domain_int] = 0 

    for domain_name in param_handler.domains:
        net = net_dict[domain_name]
        net.eval()
        loader = loader_dict[domain_name]
        domain_int = current_state.get_domain_names_to_id()[domain_name]
        
        with torch.no_grad():
            for _, (x,y,d,l) in enumerate(loader):
                x = x.to(device, dtype=torch.float)
                y,d,l = (i.to(device, dtype=torch.long) for i in [y,d,l])
                y_pred, _ = net(x)
                labeled_set_loss, labeled_num = labeled_set_ce_loss(y_pred, y, l)

                batch_clf_loss.append(labeled_set_loss.item())
                batch_labeled_num.append(labeled_num.item())

                _, y_pred_idx = y_pred.max(dim=1)
                domain_labeled_num_dict[domain_int] += labeled_num.item()
                batch_correct_number = ((y_pred_idx == y)*(l == 1)).sum().item()
                domain_correct_labeled_num_dict[domain_int] += batch_correct_number

    loss_training = np.average(batch_clf_loss, weights = batch_labeled_num)

    return loss_training, domain_labeled_num_dict, domain_correct_labeled_num_dict


#%%

def test(model_dict, test_loader_dict, current_state):
    '''
    Parameters:
    -------------------
    domain_map: map the integers to the string names.
    metric: default performance matric is accuracy.

    Return the performance on each domain.
    The return format should be a dictionary {0: acc1, 1: acc2,..., -1:acc_total}, all the keys are int.
    The -1 means the overall accuracy, because the domain_int can only be large than 0.
    '''

    # Initialize

    loss_ce = torch.nn.CrossEntropyLoss(reduce=True)
    batch_loss = []
    batch_item_num = []
    domain_int_list = list(current_state.get_domain_ids())

    correct_num = {}
    domain_num = {}
    for domain_int in domain_int_list:
        domain_num[domain_int] = 0
        correct_num[domain_int] = 0 

    for domain_name in param_handler.domains:
        net = model_dict[domain_name]
        net.eval()
        domain_int = current_state.get_domain_names_to_id()[domain_name]

        domain_num[domain_int] = len(test_loader_dict[domain_name].dataset)
        # Statistic for each batch
        with torch.no_grad():
            for x,y,_ in test_loader_dict[domain_name]:
                x = x.to(device, dtype=torch.float)
                y = y.to(device, dtype=torch.long)

                y_pred, _ = net(x)
                loss_clf = loss_ce(y_pred, y)
                batch_loss.append(loss_clf.item())
                batch_item_num.append(len(y))
                _, y_pred_idx = y_pred.max(dim=1)

                # Here the batch only contains the current domain instance.
                batch_correct_number = (y_pred_idx == y).sum().item()
                correct_num[domain_int] += batch_correct_number

    # Calculate the accuracy among all the domains.
    accuraccy_dict = {}
    for domain_int in domain_int_list:
        accuraccy_dict[domain_int] = correct_num[domain_int]/domain_num[domain_int]
    accuraccy_dict[-1] = sum(correct_num.values())/sum(domain_num.values())
    clf_loss = np.average(batch_loss, weights = batch_item_num)

    return accuraccy_dict, clf_loss
