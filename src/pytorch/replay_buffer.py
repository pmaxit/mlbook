import random
from collections import deque
from typing import NamedTuple, Tuple

import numpy as np
import torch

class ReplayBuffer:
    """ Implements the basic experience replay mechanism 
    
    :param capacity: size of the replay buffer
    :type capacity: int
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)
        
    def push(self, inp: tuple)->None:
        """ Adds new experience to buffer
        
        :param inp: Tuple containing state, action, reward, next_state, done
        :type inp: Tuple
        :returns None
        """
        self.memory.append(inp)
        
    def sample(self, batch_size: int)->(Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):
        """ Returns randomly sampled experience from replay memory
        
        :param batch_size: Number of samples per batch
        :type batch_size: int
        :returns (Tuple composing of `state`, `action`, `reward` , `next_state`, and `done`)
        
        """
        batch = random.sample(self.memory, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return [
            torch.from_numpy(v).float()
            for v in [state, action, reward, next_state, done]
        ]
        
    def __len__(self)-> int:
        """ Gives number of experiences in buffer currently
        :returns: length of replay memory
        """
        
        return self.pos
    
        
        