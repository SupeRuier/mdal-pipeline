import torch

def labeled_set_ce_loss(y_pred, y, l):
    '''
    The input is a batch of prediction result.
    '''
    # differewnt for each model

    labeled_loss = torch.nn.CrossEntropyLoss(reduce=False)

    if l.sum()==0:
        # There isn't labeled instances in this batch
        labeled_loss = torch.tensor(0)
        labeled_num = torch.tensor(0)
    else:    
        # There are labeled instances in this batch
        # Take average among the labeled data.
        labeled_loss = labeled_loss(y_pred, y)
        labeled_loss = labeled_loss * l
        labeled_num = l.sum()
        labeled_loss = labeled_loss.sum()/l.sum()
    
    return labeled_loss, labeled_num


def calculate_entropy(y_softmax):
    entropy = - torch.sum(y_softmax * torch.log(y_softmax + 1e-6), dim = 1) # Avoid gradient explode.
    return entropy