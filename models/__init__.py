from .sdl_joint import SDL_joint
from .sdl_seperate import SDL_separate
from .dann import DANN
from .mdnet import MDNet
from .man import *
from .can import *


'''
This file includes all the models used in MDAL.
Each model should have ability to make prediction and provide sufficient information to the stratety to make selection.

There should be 5 types of models with different specific structures for different dataset. [NANN,MDN,MDNet,SDL_joint,SDL_separate]

We will unify that the input to the model should be (x,d), during the training, the input should be (x,y,d).
'''
