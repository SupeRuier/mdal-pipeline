from .model_with_parts import model_by_integral_parts
import torch.nn as nn
import torch
from models.functions import ReverseLayerF
import torch.nn.functional as F

#%%
class MAN(model_by_integral_parts):

    def __init__(self, dataset_name):
        self.net_type = 'MAN'
        self.dataset = dataset_name
        
    def add_feature_s(self, net_fs):
        self.feature_s = net_fs

    def add_feature_d_dict(self, net_fd_dict):
        self.feature_d_dict = net_fd_dict

    def add_classifier(self, classifier):
        self.classifier = classifier

    def add_discriminator(self, discriminator):
        self.discriminator = discriminator

    def state_dict(self):
        state_dict = {}
        state_dict['feature_s'] = self.feature_s.state_dict()
        state_dict['classifier'] = self.classifier.state_dict()
        state_dict['discriminator'] = self.discriminator.state_dict()

        state_dict['feature_d_dict'] = {}
        for key, value in self.feature_d_dict.items():
            state_dict['feature_d_dict'][key] = value.state_dict()
        return state_dict

    def load_state_dict(self, checkpoint_dict):
        self.feature_s.load_state_dict(checkpoint_dict['feature_s'])
        self.classifier.load_state_dict(checkpoint_dict['classifier'])
        self.discriminator.load_state_dict(checkpoint_dict['discriminator'])

        for key, value in checkpoint_dict['feature_d_dict'].items():
            self.feature_d_dict[key].load_state_dict(value)

    def get_classifier(self, domain):
        return self.classifier

    def zero_grad(self):
        self.feature_s.zero_grad()
        self.classifier.zero_grad()
        self.discriminator.zero_grad()
        for feature_d in self.feature_d_dict.values():
            feature_d.zero_grad()

    def eval(self):
        self.feature_s.eval()
        self.classifier.eval()
        self.discriminator.eval()
        for feature_d in self.feature_d_dict.values():
            feature_d.eval()

    def __call__(self, x, domain, with_feature = False):
        shared_feature = self.feature_s(x)
        domain_feature = self.feature_d_dict[domain](x)
        feature_concat = torch.cat((shared_feature, domain_feature), dim=1)
        y_pred = self.classifier(feature_concat)

        if with_feature:
            return y_pred, feature_concat
        else:
            return y_pred

    def get_last_hidden_layer_size(self):
        # get the length of the last hidden layer.
        try:
            # try to read last_hidden_layer_size at first
            return self.last_hidden_layer_embd_size
        except:
            for name, module in self.classifier.named_modules():
                if '_clfgrad' in name: 
                # The layer ends with _clfgrad is the last FC layer of the classifier.
                    last_hidden_layer_shape = module.weight.shape
                    last_hidden_layer_size = last_hidden_layer_shape[1]
                    self.last_hidden_layer_size = last_hidden_layer_size
            return self.last_hidden_layer_size

    def get_softmax_batch(self, unlabeled_batch, domain):
        raw_output_batch, _ = self(unlabeled_batch, domain) 
        softmax_confidence_batch = F.softmax(raw_output_batch.detach())

        return softmax_confidence_batch

class MAN_feature_share(nn.Module):
    '''
    MAN_feature_share uses this class.
    '''
    def __init__(self, dataset_name):
        super(MAN_feature_share, self).__init__()
        self.net_type = 'MAN_feature_share'
        self.use_joint_dataset = True
        self.dataset = dataset_name

        if self.dataset == 'digits':
            self.feature = nn.Sequential()
            self.feature.add_module('f_conv1', nn.Conv2d(3, 32, kernel_size=5))
            self.feature.add_module('f_bn1', nn.BatchNorm2d(32))
            self.feature.add_module('f_pool1', nn.MaxPool2d(2))
            self.feature.add_module('f_relu1', nn.ReLU(True))
            self.feature.add_module('f_conv2', nn.Conv2d(32, 48, kernel_size=5))
            self.feature.add_module('f_bn2', nn.BatchNorm2d(48))
            self.feature.add_module('f_drop1', nn.Dropout2d())
            self.feature.add_module('f_pool2', nn.MaxPool2d(2))
            self.feature.add_module('f_relu2', nn.ReLU(True))

    def forward(self, input_data):

        if self.dataset == 'digits':
            feature = self.feature(input_data)
            feature = feature.view(-1, 48 * 4 * 4)
        else:
            feature = None

        return feature

class MAN_feature_domain_individual(nn.Module):
    '''
    Similar to MAN_feature_share but with the different output dimension.
    '''
    def __init__(self, dataset_name):
        super(MAN_feature_domain_individual, self).__init__()
        self.net_type = 'MAN_feature_domain_individual'
        self.use_joint_dataset = True
        self.dataset = dataset_name

        if self.dataset == 'digits':
            self.feature = nn.Sequential()
            self.feature.add_module('f_conv1', nn.Conv2d(3, 32, kernel_size=5))
            self.feature.add_module('f_bn1', nn.BatchNorm2d(32))
            self.feature.add_module('f_pool1', nn.MaxPool2d(2))
            self.feature.add_module('f_relu1', nn.ReLU(True))
            self.feature.add_module('f_conv2', nn.Conv2d(32, 48, kernel_size=5))
            self.feature.add_module('f_bn2', nn.BatchNorm2d(48))
            self.feature.add_module('f_drop1', nn.Dropout2d())
            self.feature.add_module('f_pool2', nn.MaxPool2d(2))
            self.feature.add_module('f_relu2', nn.ReLU(True))

    def forward(self, input_data):

        if self.dataset == 'digits':
            feature = self.feature(input_data)
            feature = feature.view(-1, 48 * 4 * 4)
        else:
            feature = None

        return feature

class MAN_classifier(nn.Module):
    '''
    This module takes the concat feature to the class output.
    We assume the feature_s and feature_d have the same output dimension.
    '''
    def __init__(self, dataset_name):
        super(MAN_classifier, self).__init__()
        self.net_type = 'MAN_classifier'
        self.use_joint_dataset = True
        self.dataset = dataset_name

        if self.dataset == 'digits':
            self.classifier_feature_extractor = nn.Sequential()
            self.classifier_feature_extractor.add_module('c_fc1', nn.Linear(48 * 4 * 4 * 2, 100))
            self.classifier_feature_extractor.add_module('c_bn1', nn.BatchNorm1d(100))
            self.classifier_feature_extractor.add_module('c_relu1', nn.ReLU(True))
            self.classifier_feature_extractor.add_module('c_drop1', nn.Dropout())
            self.classifier_feature_extractor.add_module('c_fc2', nn.Linear(100, 100))
            self.classifier_feature_extractor.add_module('c_bn2', nn.BatchNorm1d(100))
            self.classifier_feature_extractor.add_module('c_relu2', nn.ReLU(True))

            self.classifier = nn.Sequential()
            self.classifier.add_module('c_fc3_clfgrad', nn.Linear(100, 10))

    def forward(self, input_feature):
        class_output = None
        feature = None
        
        if self.dataset == 'digits':
            feature = self.classifier_feature_extractor(input_feature)
            class_output = self.classifier(feature)

        return class_output, feature

class MAN_discriminator(nn.Module):
    def __init__(self, dataset_name):
        super(MAN_discriminator, self).__init__()
        self.net_type = 'MAN_discriminator'
        self.use_joint_dataset = True
        self.dataset = dataset_name

        if self.dataset == 'digits':
            self.domain_classifier = nn.Sequential()
            self.domain_classifier.add_module('c_fc1', nn.Linear(48 * 4 * 4, 100))
            self.domain_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
            self.domain_classifier.add_module('c_relu1', nn.ReLU(True))
            self.domain_classifier.add_module('c_drop1', nn.Dropout())
            self.domain_classifier.add_module('c_fc2', nn.Linear(100, 100))
            self.domain_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
            self.domain_classifier.add_module('c_relu2', nn.ReLU(True))
            self.domain_classifier.add_module('c_fc3', nn.Linear(100, 10))

    def forward(self, input_feature):

        reverse_feature = ReverseLayerF.apply(input_feature)
        
        if self.dataset == 'digits':
            domain_output = self.domain_classifier(reverse_feature)
        else:
            domain_output = None

        return domain_output