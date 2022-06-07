from .model_by_whole import model_by_whole
import torch.nn as nn
from models.functions import ReverseLayerF
import torch.nn.functional as F

#%%
class DANN(model_by_whole):

    def __init__(self, dataset_name):
        super(DANN, self).__init__()
        self.net_type = 'DANN'
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

            self.domain_classifier = nn.Sequential()
            self.domain_classifier.add_module('d_fc1', nn.Linear(48 * 4 * 4, 100))
            self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
            self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
            self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))


    def forward(self, input_data):
        class_output = None
        feature = None
        domain_output = None

        if self.dataset == 'digits':
            feature = self.feature(input_data)
            feature = feature.view(-1, 48 * 4 * 4)
            reverse_feature = ReverseLayerF.apply(feature)
            feature = self.classifier_feature_extractor(feature)
            class_output = self.classifier(feature)
            domain_output = self.domain_classifier(reverse_feature)

        return class_output, feature, domain_output

    def __call__(self, x, require_domain_output = False):
        class_output, feature, domain_output = self.forward(x)
        if require_domain_output == True:
            return class_output, feature, domain_output
        else:
            return class_output, feature

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

    def get_softmax_batch(self, unlabeled_batch):
        raw_output_batch, _ = self(unlabeled_batch) 
        softmax_confidence_batch = F.softmax(raw_output_batch.detach())

        return softmax_confidence_batch