from .model_with_parts import model_by_integral_parts
import torch.nn as nn
import torch.nn.functional as F

#%%
class MDNet(model_by_integral_parts):

    def __init__(self, dataset_name):
        self.net_type = 'MDNet'
        self.dataset = dataset_name
        
    def add_feature_net(self, net_f):
        self.feature_net = net_f

    def add_clf_net_dict(self, clf_net_dict):
        self.clf_net_dict = clf_net_dict

    def state_dict(self):
        state_dict = {}
        state_dict['feature_net'] = self.feature_net.state_dict()
        state_dict['clf_net_dict'] = {}
        for key, value in self.clf_net_dict.items():
            state_dict['clf_net_dict'][key] = value.state_dict()
        return state_dict

    def load_state_dict(self, checkpoint_dict):
        self.feature_net.load_state_dict(checkpoint_dict['feature_net'])
        for key, value in checkpoint_dict['clf_net_dict'].items():
            self.clf_net_dict[key].load_state_dict(value)

    def get_classifier(self, domain):
        return self.clf_net_dict[domain]

    def zero_grad(self):
        self.feature_net.zero_grad()
        for clf_net in self.clf_net_dict.values():
            clf_net.zero_grad()

    def eval(self):
        self.feature_net.eval()
        for clf_net in self.clf_net_dict.values():
            clf_net.eval()

    def __call__(self, x, domain):
        return self.clf_net_dict[domain](self.feature_net(x))

    def get_last_hidden_layer_size(self):
        # get the length of the last hidden layer.
        try:
            # try to read last_hidden_layer_size at first
            return self.last_hidden_layer_embd_size
        except:
            for name, module in list(self.clf_net_dict.values())[0].named_modules():
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

class MDNet_feature(nn.Module):
    def __init__(self, dataset_name):
        super(MDNet_feature, self).__init__()
        self.net_type = 'MDNet_feature'
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

class MDNet_classifier_individual(nn.Module):
    def __init__(self, dataset_name):
        super(MDNet_classifier_individual, self).__init__()
        self.net_type = 'MDNet_classifier_individual'
        self.use_joint_dataset = True
        self.dataset = dataset_name

        if self.dataset == 'digits':
            self.classifier_feature_extractor = nn.Sequential()
            self.classifier_feature_extractor.add_module('c_fc1', nn.Linear(48 * 4 * 4, 100))
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

