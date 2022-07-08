import math
import random
from copy import deepcopy
from typing import Any, Dict, List, Tuple

import torch  # noqa
import torch.optim as opt  # noqa
import torch.nn as nn  # noqa
import numpy as np


from genrl.agents import OffPolicyAgent
from genrl.utils import get_env_properties, get_model, safe_mean
from genrl.environments import VectorEnv
from genrl.trainers import OffPolicyTrainer
import gym

def get_env_properties(
    env, network = 'mlp'
):
    """ Finds important properties of environment 
    :param env: Environment that the agent is interacting with
    :type env: Gym environment
    :param environment: Type of network architecture e.g. mlp, cnn
    :type env: str
    :returns (state space dimension, Action space dimension)
    """
    
    if network == 'mlp':
        state_dim = env.observation_space.shape[0]
 
    if isinstance(env.action_space, gym.spaces.Discrete):
        action_dim = env.action_space.n
        discrete = True
        action_lim = None
    elif isinstance(env.action_space, gym.spaces.Box):
        action_dim = env.action_space.shape[0]
        action_lim = env.action_space.high[0]
        discrete = False
    else: 
        raise NotImplementedError
    
    return state_dim, action_dim, discrete, action_lim

def mlp(
    sizes: Tuple,
    activation: str='relu',
    sac:bool = False
):
    """ Generates an MLP model given sizes of each layer"""
    layers = []
    limit = len(sizes) if sac is False else len(sizes)-1
    activation = nn.Tanh() if activation == "tanh" else nn.ReLU()

    for layer in range(limit - 1):
        act = activation if layer < limit - 2 else nn.Identity()
        layers += [nn.Linear(sizes[layer], sizes[layer + 1]), act]

    return nn.Sequential(*layers)    


class DQN(OffPolicyAgent):
    """ Base DQN class """
    def __init__(self, *args, max_epsilon: float = 1.0, min_epsilon=0.01, epsilon_decay:int = 500, **kwargs):
        super(DQN, self).__init__(*args, **kwargs)
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.dqn_type = ""
        self.noisy = False
        
        self.empty_logs()
        
        self._create_model()
        
        
    def _create_model(self, *args, **kwargs)-> None:
        """ FUnction to initialize Q-value model"""
        
        state_dim, action_dim, discrete, _ = get_env_properties(self.env, self.network)
        fc_layers = (32, 32)
        # get state dimension 
        self.model = mlp([state_dim, *fc_layers, action_dim], activation='relu')
        self.target_model = deepcopy(self.model)
        
        self.optimizer = opt.Adam(self.model.parameters(), lr = self.lr_value)
    
    def update_target_model(self) -> None:
        """Function to update the target Q model
        Updates the target model with the training model's weights when called
        """
        self.target_model.load_state_dict(self.model.state_dict())
        
    
    def update_params_before_select_action(self, timestep: int) -> None:
        """Update necessary parameters before selecting an action
        This updates the epsilon (exploration rate) of the agent every timestep
        Args:
            timestep (int): Timestep of training
        """
        self.timestep = timestep
        self.epsilon = self.calculate_epsilon_by_frame()
        self.logs["epsilon"].append(self.epsilon)
    
    def get_greedy_action(self, state: torch.Tensor) -> torch.Tensor:
        """Greedy action selection
        Args:
            state (:obj:`torch.Tensor`): Current state of the environment
        Returns:
            action (:obj:`torch.Tensor`): Action taken by the agent
        """
        q_values = self.model(state.unsqueeze(0))
        action = torch.argmax(q_values.squeeze(), dim=-1)
        return action
    
    def select_action(
        self, state: torch.Tensor, deterministic: bool = False
    ) -> torch.Tensor:
        """Select action given state
        Epsilon-greedy action-selection
        Args:
            state (:obj:`torch.Tensor`): Current state of the environment
            deterministic (bool): Should the policy be deterministic or stochastic
        Returns:
            action (:obj:`torch.Tensor`): Action taken by the agent
        """
        action = self.get_greedy_action(state)
        if not deterministic:
            if random.random() < self.epsilon:
                action = self.env.sample()
        return action
    
    def _reshape_batch(self, batch: List):
        """Function to reshape experiences for DQN
        Most of the DQN experiences need to be reshaped before sending to the
        Neural Networks
        """
        states = batch[0]
        actions = batch[1].unsqueeze(-1).long()
        rewards = batch[2]
        next_states = batch[3]
        dones = batch[4]
        
        return states, actions, rewards, next_states, dones
    
    def get_q_values(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Get Q values corresponding to specific states and actions
        Args:
            states (:obj:`torch.Tensor`): States for which Q-values need to be found
            actions (:obj:`torch.Tensor`): Actions taken at respective states
        Returns:
            q_values (:obj:`torch.Tensor`): Q values for the given states and actions
        """
        q_values = self.model(states)
        q_values = q_values.gather(2, actions)
        return q_values
    
    def get_target_q_values(self, next_states: torch.Tensor, rewards: List[float], dones: List[bool]):
        """ Get target Q values for the DQN"""
        
        next_q_target_values = self.target_model(next_states)
        # maximum of next q target values
        max_next_q_target_values = next_q_target_values.max(2)[0]
        target_q_values = rewards + self.gamma * torch.mul(
            max_next_q_target_values, (1-dones)
        )
        
        return target_q_values.unsqueeze(-1)
    

    def update_params(self, update_interval: int) -> None:
        """Update parameters of the model
        Args:
            update_interval (int): Interval between successive updates of the target model
        """
        self.update_target_model()
        for timestep in range(update_interval):
            batch = self.sample_from_buffer()
            loss = self.get_q_loss(batch)
            
            self.logs["value_loss"].append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # In case the model uses Noisy layers, we must reset the noise every timestep
            if self.noisy:
                self.model.reset_noise()
                self.target_model.reset_noise()
                
    def calculate_epsilon_by_frame(self) -> float:
        """Helper function to calculate epsilon after every timestep
        Exponentially decays exploration rate from max epsilon to min epsilon
        The greater the value of epsilon_decay, the slower the decrease in epsilon
        """
        return self.min_epsilon + (self.max_epsilon - self.min_epsilon) * math.exp(
            -1.0 * self.timestep / self.epsilon_decay
        )

    def get_hyperparams(self) -> Dict[str, Any]:
        """Get relevant hyperparameters to save
        Returns:
            hyperparams (:obj:`dict`): Hyperparameters to be saved
            weights (:obj:`torch.Tensor`): Neural network weights
        """
        hyperparams = {
            "gamma": self.gamma,
            "batch_size": self.batch_size,
            "lr": self.lr_value,
            "replay_size": self.replay_size,
            "weights": self.model.state_dict(),
            "timestep": self.timestep,
        }
        return hyperparams, self.model.state_dict()

    def load_weights(self, weights) -> None:
        """Load weights for the agent from pretrained model
        Args:
            weights (:obj:`torch.Tensor`): neural net weights
        """
        self.model.load_state_dict(weights)
        
    def get_logging_params(self) -> Dict[str, Any]:
        """Gets relevant parameters for logging
        Returns:
            logs (:obj:`dict`): Logging parameters for monitoring training
        """
        logs = {
            "value_loss": safe_mean(self.logs["value_loss"]),
            "epsilon": safe_mean(self.logs["epsilon"]),
        }
        #print(logs)
        self.empty_logs()
        return logs

    def empty_logs(self) -> None:
        """Empties logs"""
        self.logs = {}
        self.logs["value_loss"] = []
        self.logs["epsilon"] = []
        
if __name__ == '__main__':
    env = VectorEnv("CartPole-v0")
    agent = DQN("mlp", env)
    trainer = OffPolicyTrainer(agent, env, max_timesteps=20000,epochs=20000)
    
    trainer.train()
    trainer.evaluate(render=False)
    
    # save the model
    #trainer.save()
