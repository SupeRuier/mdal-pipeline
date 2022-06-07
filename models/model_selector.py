from .sdl_joint import SDL_joint
from .sdl_seperate import *
from .dann import DANN
from .mdnet import *
from .man import *
from .can import *

def model_selector(model_name, dataset_name):
    selected_model = None
    if model_name == 'DANN':
        selected_model = DANN(dataset_name)
    elif model_name == 'SDL_joint':
        selected_model = SDL_joint(dataset_name)
    elif model_name == 'SDL_separate':
        selected_model = SDL_separate(dataset_name)
    elif model_name == 'SDL_separate_individual':
        selected_model = SDL_separate_individual(dataset_name)
    elif model_name == 'MDNet':
        selected_model = MDNet(dataset_name)
    elif model_name == 'MDNet_feature':
        selected_model = MDNet_feature(dataset_name)
    elif model_name == 'MDNet_classifier_individual':
        selected_model = MDNet_classifier_individual(dataset_name)
    elif model_name == 'MAN':
        selected_model = MAN(dataset_name)
    elif model_name == 'MAN_feature_share':
        selected_model = MAN_feature_share(dataset_name)
    elif model_name == 'MAN_discriminator':
        selected_model = MAN_discriminator(dataset_name)
    elif model_name == 'MAN_classifier':
        selected_model = MAN_classifier(dataset_name)
    elif model_name == 'MAN_feature_domain_individual':
        selected_model = MAN_feature_domain_individual(dataset_name)
    elif model_name == 'CAN':
        selected_model = CAN(dataset_name)
    elif model_name == 'CAN_feature_share':
        selected_model = CAN_feature_share(dataset_name)
    elif model_name == 'CAN_discriminator':
        selected_model = CAN_discriminator(dataset_name)
    elif model_name == 'CAN_classifier':
        selected_model = CAN_classifier(dataset_name)
    elif model_name == 'CAN_feature_domain_individual':
        selected_model = CAN_feature_domain_individual(dataset_name)

    return selected_model