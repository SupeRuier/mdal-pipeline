from .model_with_parts import model_by_integral_parts
from .sdl_joint import SDL_joint
import torch.nn.functional as F

#%%
class SDL_separate(model_by_integral_parts):

    def __init__(self, dataset_name):
        self.net_type = 'SDL_separate'
        self.dataset = dataset_name
        
    def add_individual_dict(self, individual_dict):
        self.individual_dict = individual_dict

    def state_dict(self):
        state_dict = {}
        for key, value in self.individual_dict.items():
            state_dict[key] = value.state_dict()
        return state_dict

    def load_state_dict(self, checkpoint_dict):
        for key, value in checkpoint_dict.items():
            self.individual_dict[key].load_state_dict(value)

    def get_classifier(self, domain):
        return self.individual_dict[domain]

    def zero_grad(self):
        for individual_model in self.individual_dict.values():
            individual_model.zero_grad()

    def eval(self):
        for individual_model in self.individual_dict.values():
            individual_model.eval()

    def __call__(self, x, domain):
        return self.individual_dict[domain](x)

    def get_last_hidden_layer_size(self):
        # get the length of the last hidden layer.
        try:
            # try to read last_hidden_layer_size at first
            return self.last_hidden_layer_embd_size
        except:
            for name, module in list(self.individual_dict.values())[0].named_modules():
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

class SDL_separate_individual(SDL_joint):
    '''
    This is just a normal NN.
    Same structure with SDL_joint.
    '''
    def __init__(self, dataset_name):
        super(SDL_separate_individual, self).__init__(dataset_name)
        self.net_type = 'SDL_separate_individual'
        self.dataset = dataset_name
