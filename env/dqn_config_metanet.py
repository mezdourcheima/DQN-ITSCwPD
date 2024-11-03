from .custom_env import SUMO_PARAMS
import torch
import torch.nn as nn
import torch.optim as optim
from torch import no_grad, as_tensor
from gym.spaces import Box

#CONFIG = SUMO_PARAMS["config"]
#Config for metanet model
CONFIG = "J7_TLS/metanet"
# """CHANGE HYPER PARAMETERS HERE""" ###################################################################################
HYPER_PARAMS = {
    'gpu': '1',                                 # GPU #
    'n_env': 1,                                 # Multi-processing environments
    'lr': 0.0001,                                # Learning rate - Increased
    'gamma': 0.99,                              # Discount factor
    'eps_start': 1.0,                           # Epsilon start
    'eps_min': 0.01,                            # Epsilon min
    'eps_dec': 1e6,                             # Epsilon decay - Reduced
    'eps_dec_exp': False,                       # Epsilon exponential decay - Switched to linear decay
    'bs': 64,                                   # Batch size - Increased
    'min_mem': 100000,                          # Replay memory buffer min size
    'max_mem': 1000000,                         # Replay memory buffer max size
    'target_update_freq': 30000,                # Target network update frequency
    'target_soft_update': True,                 # Target network soft update
    'target_soft_update_tau': 5e-03,            # Target network soft update tau rate - Increased
    'save_freq': 10000,                         # Save frequency
    'log_freq': 1000,                           # Log frequency
    'save_dir': './save/' + CONFIG + "/",       # Save directory
    'log_dir': './logs/train/' + CONFIG + "/",  # Log directory
    'load': True,                               # Load model
    'repeat': 0,                                # Repeat action
    'max_episode_steps': 800,                   # Time limit episode steps - Reduced
    'max_total_steps': 0,                       # Max total training steps if > 0, else inf training
    'algo': 'DuelingDoubleDQNAgent'             # DQNAgent
                                                # DoubleDQNAgent
                                                # DuelingDoubleDQNAgent
                                                # PerDuelingDoubleDQNAgent
                                                # AlineaAgent
}

########################################################################################################################


# """CHANGE NETWORK CONFIG HERE""" #####################################################################################
def network_config(input_dim):
    # """CHANGE NETWORK HERE""" ########################################################################################
    """"""
    
    
    cnn_dims = (
        (16, 1, 1),
        (32, 1, 1)
    )

    fc_dims = (128, 64)

    activation = nn.ELU()

    cnn = nn.Sequential(
        nn.Conv2d(input_dim.shape[0], cnn_dims[0][0], kernel_size=cnn_dims[0][1], stride=cnn_dims[0][2]),
        activation,
        nn.Conv2d(cnn_dims[0][0], cnn_dims[1][0], kernel_size=cnn_dims[1][1], stride=cnn_dims[1][2]),
        activation,
        nn.Flatten()
    )

    with no_grad():
        n_flatten = cnn(as_tensor(input_dim.sample()[None]).float()).shape[1]


    net = nn.Sequential(
        cnn,
        nn.Linear(n_flatten, fc_dims[0]),
        activation,
        nn.Linear(fc_dims[0], fc_dims[1]),
        activation
    )
    """"""
    """
    cnn_dims = (
        (32, 4, 2),
        (64, 2, 2),
        (128, 2, 1)
    )

    fc_dims = (128, 64)

    activation = nn.ELU()

    cnn = nn.Sequential(
        nn.Conv2d(input_dim.shape[0], cnn_dims[0][0], kernel_size=cnn_dims[0][1], stride=cnn_dims[0][2]),
        activation,
        nn.Conv2d(cnn_dims[0][0], cnn_dims[1][0], kernel_size=cnn_dims[1][1], stride=cnn_dims[1][2]),
        activation,
        nn.Conv2d(cnn_dims[1][0], cnn_dims[2][0], kernel_size=cnn_dims[2][1], stride=cnn_dims[2][2]),
        activation,
        nn.Flatten()
    )

    with no_grad():
        n_flatten = cnn(as_tensor(input_dim.sample()[None]).float()).shape[1]

    net = nn.Sequential(
        cnn,
        nn.Linear(n_flatten, fc_dims[0]),
        activation,
        nn.Linear(fc_dims[0], fc_dims[1]),
        activation
    )
    """
    ####################################################################################################################

    # """CHANGE FC DUELING LAYER OUTPUT DIM HERE""" ####################################################################
    fc_out_dim = fc_dims[-1]
    ####################################################################################################################

    # """CHANGE OPTIMIZER HERE""" ######################################################################################
    optim_func = optim.Adam
    ####################################################################################################################

    # """CHANGE LOSS HERE""" ###########################################################################################
    loss_func = nn.SmoothL1Loss
    ####################################################################################################################

    return net, fc_out_dim, optim_func, loss_func

########################################################################################################################
