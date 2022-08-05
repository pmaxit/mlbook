from typing import List, Type, Union

import numpy as np
from replay_buffer import ReplayBuffer
from genrl.trainers.base import Trainer

class OffPolicyTrainer(Trainer):
    """ OffPolicyTrainer class
        Trainer class for all the Off Policy Agents: DQN (all variants), DDPG, TD3 and SAC

   Attributes:
        agent (object): Agent algorithm object
        env (object): Environment
        buffer (object): Replay Buffer object
        max_ep_len (int): Maximum Episode length for training
        max_timesteps (int): Maximum limit of timesteps to train for
        warmup_steps (int): Number of warmup steps. (random actions are taken to add randomness to training)
        start_update (int): Timesteps after which the agent networks should start updating
        update_interval (int): Timesteps between target network updates
        log_mode (:obj:`list` of str): List of different kinds of logging. Supported: ["csv", "stdout", "tensorboard"]
        log_key (str): Key plotted on x_axis. Supported: ["timestep", "episode"]
        log_interval (int): Timesteps between successive logging of parameters onto the console
        logdir (str): Directory where log files should be saved.
        epochs (int): Total number of epochs to train for
        off_policy (bool): True if the agent is an off policy agent, False if it is on policy
        save_interval (int): Timesteps between successive saves of the agent's important hyperparameters
        save_model (str): Directory where the checkpoints of agent parameters should be saved
        run_num (int): A run number allotted to the save of parameters
        load_model (str): File to load saved parameter checkpoint from
        render (bool): True if environment is to be rendered during training, else False
        evaluate_episodes (int): Number of episodes to evaluate for
        seed (int): Set seed for reproducibility
    """
    
    