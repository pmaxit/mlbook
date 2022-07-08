#!/usr/bin/env python
# coding: utf-8

# # OpenAI gym
# 
# OpenAI is an **artificial intlligence (AI)** research organization that aims to build **artificial general intelligence (AGI)**. OpenAI provides a fmaous toolkit called Gym for training a reinforcement learning agent.

# ## Creating our first Gym environment

# Let's introduce one of the simplest environments called Frozen Lake environment. As we can observe, in the frozen environment, the goal of the agent is to start from the initial state S and reach the goal state **G**

# - S denotes the starting state
# - F denotes the frozen state
# - H denotes the hole state
# - G denotes the goal state

# ### Goal
# 
# Goal of the agent is to start from state **S** and reach the goal state **G** without touching **H**. It can only travel over F

# ## Environment

# In[1]:


import gym


# In[2]:


env = gym.make('FrozenLake-v1')
env.reset()


# In[ ]:





# In[3]:


env.render()


# ## Exploring the environment
# 
# In the previous chapter, we learned that the reinforcement learning environment can be modeled as **Markov Decision Process (MDP)** and an MDP consists of the following:
# 
# - **States**
# - **Actions**
# - **Transitino probability**
# - **Reward function**

# ### States
# 
# A state space consists of all of our statgse. We can obtain the number of states in our environment as below

# In[4]:


print(env.observation_space)


# It implies that we have 16 discrete states in our state space starting from states **S** to **G**.

# ### Actions
# 
# We learned that action space consts of all possible actions. In our case we have 4 discrete actions in our action space, which are **left, down, right and up**.

# ### Transition probability

# It is a stochastic environment, we cannot say that by performing some action **a**, the agent will always reach the enxt state **s** We also reach other states with some probability. So when we perform an action 1 (down) in state 2, we reach state 1 with probability (0.33), we reach state 6 with probability 0.333 and we reach state 3 with the same probability.

# In[5]:


env.P[2][0]


# Our output is in the form of (transition probability, next state, reward, is Terminal )

# ## Generating an episode

# In order for an agent to interact with the environment, it has to perform some action in the environment.

# In[6]:


env.step(1)


# In[7]:


env.render()


# Episode is the agent environment interaction startin from initial state to terminal state. An episode ends if the agent reaches the terminal state. So, in the frozen lake environment, the episode will end if agent reaches the terminal state.
# 
# Let's understand how to generate an episode with the random policy. We learned that the random policy selects a random action in each state. So we will generate an episode by taking random actions.

# In[8]:


num_episodes = 10
num_timesteps = 20


# In[9]:


for i in range(num_episodes):
    print('Episode # ', i)
    state = env.reset()
    for t in range(num_timesteps):
        
        random_action = env.action_space.sample()
        next_state, reward, done, info = env.step(random_action)
        print('time step', t+1)
        env.render()
        if done:
            break


# In[ ]:




