#!/usr/bin/env python
# coding: utf-8

# # Pytorch for reinforcement learning

# Here is teh quick introduction to reinforcement learning with pytorch

# In[1]:


import pfrl
import torch
import torch.nn
import gym
import numpy


# PFRL can be used for any problems if they are modeled as "enviroments". Open AI gym provides various kinds of benchmark environ ments and defined scommon interface among them.

# In[2]:


env = gym.make('CartPole-v0')
print('observation space:', env.observation_space)
print('action space:', env.action_space)

obs = env.reset()
print('initial observation:', obs)

action = env.action_space.sample()
obs, r, done, info = env.step(action)
print('next observation:', obs)
print('reward:', r)
print('done:', done)
print('info:', info)

# Uncomment to open a GUI window rendering the current state of the environment
# env.render()


# PFRL provides various agents, each of which implements a deep reinforcement learning aglrithm.
# 
# Let's try to use DoubleDQN algorithm which is implemented by `pfrl.agents.DoubleDQN`. this algorithm trains a Q-function that receives an observation and returns an expected future return for each action that agent can take. You an define your Q-function as `torch.nn.Module` as below.

# In[3]:


class QFunction(torch.nn.Module):
    def __init__(self, obs_size, n_actions):
        super().__init__()
        self.l1 = torch.nn.Linear(obs_size, 50)
        self.l2 = torch.nn.Linear(50, 50)
        self.l3 = torch.nn.Linear(50, n_actions)
        
    def forward(self, x):
        h = x
        h = torch.nn.functional.relu(self.l1(h))
        h = torch.nn.functional.relu(self.l2(h))
        h = self.l3(h)
        
        return pfrl.action_value.DiscreteActionValue(h)
    
obs_size = env.observation_space.low.size
n_actions = env.action_space.n
q_function = QFunction(obs_size, n_actions)


# `pfrl.q_fuctions.DiscrenteActionValueHead` is just a torch.nn.Module that packs ints input to `pfrl.action_value.DiscreteActionValue`

# As usual in PyTorch, `torch.optim.Optimizer` is used to optimize a model
# 

# In[4]:


# Use Adam to optimize q_func. eps=1e-2 is for stability.
optimizer = torch.optim.Adam(q_function.parameters(), eps=1e-2)


# # Create Agent

# In[5]:


gamma = 0.9

# use epsilon-greedy for exploration
explorer = pfrl.explorers.ConstantEpsilonGreedy(epsilon=0.3, random_action_func= env.action_space.sample)

# DQN uses experience replay
replay_buffer = pfrl.replay_buffers.ReplayBuffer(capacity=10**6)


# since the observations from CartPole-v0 is numpy.float64 
# whie as pytorch only accepts numpy.float32 by default, specify
# a converter as a feature extractor function phi
phi = lambda x: x.astype(numpy.float32, copy=False)

gpu = -1

agent = pfrl.agents.DoubleDQN(
    q_function ,
    optimizer,
    replay_buffer,
    gamma,
    explorer,
    replay_start_size=500,
    update_interval=1,
    target_update_interval=100,
    phi=phi,
    gpu=gpu
)


# Now that you have an agent and an environment, it's time to start reinforcement learning!

# During training, two methods of `agent` must be called: `agent.act` and `agent.observe` 
# 
# * `agent.act(obs)` takes the curernt observation as input and returns an exploraty action. Once the returned action is processed in env, 
# 
# * `agent.observe(obs, reeward, done, reset)` then observes the consequences
# 
# - `obs` : next observation
# - `reward` : an immediate reward
# - `done` : a boolean value set to True if reached a terminal state
# - `reset` : a boolean value set to True if an episode is interrupted at a non-terminal state, typically by a time limit
# 

# In[6]:


n_episodes = 300
max_episode_len = 200

history = []
for i in range(1 , n_episodes+1):
    obs = env.reset()
    R = 0
    t = 0
    
    while True:
        # Uncomment to watch the behavior in GUI window
        
        action = agent.act(obs)
        obs, reward, done, _ = env.step(action)
        R += reward
        t += 1
        
        reset = t == max_episode_len
        agent.observe(obs, reward, done, reset)
        if done or reset:
            history.append(R)
            break
        
    
    if i %10 == 0:
        print('episode : ',i ,'R: ', R)
        
    if i % 50 == 0:
        print('episode : ', agent.get_statistics())


# Now you finished the training the Double DQN agent for 300 episodes. How good is th agent now? You can evaluate it using `with agent.eval_mode()` . Exploration such as epsilon-greedy is not used anymore.

# In[7]:


with agent.eval_mode():
    for i in range(10):
        obs = env.reset()
        R = 0
        t = 0
        while True:
            action = agent.act(obs)
            obs, r , done , _ = env.step(action)
            
            R += r
            t += 1
            
            reset = t== 200
            agent.observe(obs, r, done, reset)
            if done or reset:
                break
        
        print('evaluation episode: ', i, 'R: ', R)


# # Finishing up

# In[8]:


agent.save('agent')


# # Shortcut

# In[9]:


# Set up the logger to print info messages for understandability.
import logging
import sys
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='')

pfrl.experiments.train_agent_with_evaluation(
    agent,
    env,
    steps=2000,           # Train the agent for 2000 steps
    eval_n_steps=None,       # We evaluate for episodes, not time
    eval_n_episodes=10,       # 10 episodes are sampled for each evaluation
    train_max_episode_len=200,  # Maximum length of each episode
    eval_interval=1000,   # Evaluate the agent after every 1000 steps
    outdir='result',      # Save everything to 'result' directory
)


# # Rainbow DQN

# In[10]:


from pfrl.q_functions import DistributionalDuelingDQN


# In[11]:


n_atoms = 51
v_max = 10
v_min = -10

q_func = q_functions.DistributionalFCStateQFunctionWithDiscreteAction(
        obs_size,
        n_actions,
        n_atoms,
        v_min,
        v_max,
        n_hidden_channels=args.n_hidden_channels,
        n_hidden_layers=args.n_hidden_layers,
)print(q_func)


# In[ ]:




