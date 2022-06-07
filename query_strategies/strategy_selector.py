from .random import RandomStrategy
from .uncertainty import Uncertainty
from .expected_gradient_length import EGL
from .badge import BADGE
from .coreset import Coreset

def strategy_selector(strategy_name):

    if strategy_name == 'Random':
        selected_strategy = RandomStrategy()
    elif strategy_name == 'Uncertainty':
        selected_strategy = Uncertainty()
    elif strategy_name == 'EGL':
        selected_strategy = EGL()
    elif strategy_name == 'BADGE':
        selected_strategy = BADGE()    
    elif strategy_name == 'Coreset':
        selected_strategy = Coreset()
    else:
        selected_strategy = None

    return selected_strategy