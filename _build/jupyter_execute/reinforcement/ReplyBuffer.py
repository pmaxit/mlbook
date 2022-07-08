#!/usr/bin/env python
# coding: utf-8

# # Replay Buffer
# 
# Replay Buffer to store trajectories of experience when executing a policy in an environment. During training, replay buffers are queried for a subset of the trajectories ( either a sequential subset or a sample ) to "replay" the agent's exeprience

# In[1]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from tf_agents import specs
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.networks import q_network
from tf_agents.replay_buffers import py_uniform_replay_buffer
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step, Trajectory


# # Environment

# In[2]:


env_name = "CartPole-v0" # @param {type:"string"}
num_iterations = 250 # @param {type:"integer"}
collect_episodes_per_iteration = 2 # @param {type:"integer"}
replay_buffer_capacity = 2000 # @param {type:"integer"}

fc_layer_params = (100,)

learning_rate = 1e-3 # @param {type:"number"}
log_interval = 25 # @param {type:"integer"}
num_eval_episodes = 10 # @param {type:"integer"}
eval_interval = 50 # @param {type:"integer"}


# In[3]:


train_env = suite_gym.load(env_name)
eval_env = suite_gym.load(env_name)

train_tf_env = tf_py_environment.TFPyEnvironment(train_env)
eval_tf_env = tf_py_environment.TFPyEnvironment(eval_env)


# It has two elements, `time_step_spec` and `action_spec`. These two will go to trajectory

# In[4]:


print('action_spec:', train_env.action_spec())
print(train_env.time_step_spec())


# We see that environment expects action of type `int64` and rturns `TimeSteps` where observation are a `float32` vector of length 4 and discount factor is `float32`. now let's try to take a fixed action `(1,)`

# In[5]:


action = np.array(1, dtype=np.int32)
time_step = train_env.reset()
print(time_step)
while not time_step.is_last():
    time_step = train_env.step(action)
    print(time_step)


# # Store the transitions

# In[6]:


time_step = train_env.reset()
rewards = []
steps = []
num_episodes = 5
for _ in range(num_episodes):
    episode_reward = 0
    episode_steps = 0
    while not time_step.is_last():
        action = np.random.choice([0,1])
        time_step = train_env.step(action)
        episode_steps += 1
        episode_reward += time_step.reward
        
    rewards.append(episode_reward)
    steps.append(episode_steps)
    time_step = train_env.reset()
    
num_steps = np.sum(steps)
avg_length = np.mean(steps)
avg_reward = np.mean(rewards)


# In[7]:


print('num_episodes:', num_episodes, 'num_steps:', num_steps)
print('avg_length', avg_length, 'avg_reward:', avg_reward)


# # Replay Buffers

# `PyUniformReplayBuffers` can be used to store the episodes and convert to batch.

# For most agents, `collect_data_spec` is a named tuple called `Trajectory`, containing the specs for observations, actions, rewards, and other items."""
# 
# agent.collect_data_spec
# 
# agent.collect_data_spec._fields
# 
# ## Data Collection

# The most important method is action(time_step) which maps a time_step containing an observation from the environment to a PolicyStep named tuple containing the following attributes:

# * action: The action to be applied to the environment.
# * state: The state of the policy (e.g. RNN state) to be fed into the next call to action.
# * info: Optional side information such as action log probabilities.

# Environment provides two spec 
# * env.time_step_spec() 
# * env.action_spec()

# In[8]:


replay_buffer_capacity = 1000*32


# In[9]:


from tf_agents.agents import data_converter
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.utils import test_utils
from tf_agents.trajectories import policy_step
from tf_agents.specs import array_spec
from tf_agents.utils import nest_utils

from tf_agents.drivers import py_driver
from tf_agents.policies import py_tf_eager_policy
from tf_agents.replay_buffers import episodic_replay_buffer

from tf_agents.drivers import dynamic_episode_driver


# In[10]:


time_step_spec = train_env.time_step_spec()
action_spec = policy_step.PolicyStep(train_env.action_spec())


# In[11]:


data_spec = trajectory.from_transition(time_step_spec, action_spec, time_step_spec)


# In[12]:


py_replay_buffer = episodic_replay_buffer.EpisodicReplayBuffer(
    capacity=2,
    data_spec=data_spec,
    completed_only=True)

stateful_buffer = episodic_replay_buffer.StatefulEpisodicReplayBuffer(
    py_replay_buffer,
    num_episodes=2,
)


# In[ ]:


def collect_step(environment, policy, buffer, id, policy_state):
  time_step = environment.current_time_step()
  if policy_state:
      action_step = policy.action(time_step, policy_state)
      policy_state = action_step.state
  else:
      action_step = policy.action(time_step)

  next_time_step = environment.step(action_step.action)
  traj = trajectory.from_transition(time_step, action_step, next_time_step)

  id_tensor = tf.constant(id, dtype=tf.int64)
  buffer.add_batch(traj, id_tensor)
  if time_step.is_last():
      id[0] += 1

  return policy_state


# In[ ]:


def collect_data(env, policy, buffer, steps, id, policy_state=()):
  for _ in range(steps):
    policy_state = collect_step(env, policy, buffer, id, policy_state)

  return policy_state


# In[ ]:


episode_id = [0]
collect_data(train_env, random_policy, py_replay_buffer, 2 , episode_id)


# In[ ]:





# In[ ]:




def collect_episode(environment, policy, num_episodes):

  driver = py_driver.PyDriver(
    environment,
    py_tf_eager_policy.PyTFEagerPolicy(
      policy, use_tf_function=True),
    [py_replay_buffer.add_batch],
    max_episodes=num_episodes)
  initial_time_step = environment.reset()
  driver.run(initial_time_step)


# In[ ]:


collect_episode(train_env, tf_agent.collect_policy, 2)


# In[ ]:


def collect_episode(environment, policy, buffer=None, steps=2):
    observers = [buffer.add_batch]
    
    driver = dynamic_episode_driver.DynamicEpisodeDriver(
            environment, policy, observers,num_episodes=steps)
    
    final_time_step, policy_state = driver.run()
    


# In[ ]:


#@test {"skip": true}
def collect_step(environment, policy, buffer):
  time_step = environment.current_time_step()
  action_step = policy.action(time_step)
  next_time_step = environment.step(action_step.action)
  traj = trajectory.from_transition(time_step, action_step, next_time_step)

  #print(traj)
  # Add trajectory to the replay buffer
  buffer.add_batch(nest_utils.batch_nested_array(traj))


# In[ ]:


from tf_agents.policies import random_py_policy


# In[ ]:


random_policy = random_py_policy.RandomPyPolicy(train_env.time_step_spec(),
                                                train_env.action_spec())


# In[ ]:


def collect_data(env, policy, buffer, steps):
  for _ in range(steps):
    collect_step(env, policy, buffer)

collect_data(train_env, random_policy, py_replay_buffer, steps=100)


# In[ ]:


collect_episode(train_tf_env, random_policy, py_replay_buffer, steps=5)


# In[ ]:


dataset = py_replay_buffer.as_dataset(
    sample_batch_size=batch_size,
    num_steps=n_step_update + 1).prefetch(3)

iterator = iter(dataset)


# # Training

# In[ ]:


def compute_avg_return(environment, policy, num_episodes = 10):
    total_return = 0.0
    for _ in range(num_episodes):
        time_step = environment.reset()
        episode_return = 0.0
        
        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
            
        total_return += episode_return
        
    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


# In[ ]:


train_tf_env = tf_py_environment.TFPyEnvironment(train_env)
eval_tf_env = tf_py_environment.TFPyEnvironment(eval_env)


# In[ ]:


from tf_agents.networks import actor_distribution_network
from tf_agents.policies import py_tf_eager_policy


# In[ ]:



fc_layer_params = (100,)


# In[ ]:


actor_net = actor_distribution_network.ActorDistributionNetwork(
    train_tf_env.observation_spec(),
    train_tf_env.action_spec(),
    fc_layer_params=fc_layer_params)


# In[ ]:


from tf_agents.agents.reinforce import reinforce_agent


# In[ ]:


optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

train_step_counter = tf.Variable(0)

tf_agent = reinforce_agent.ReinforceAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    actor_network=actor_net,
    optimizer=optimizer,
    normalize_returns=True,
    train_step_counter=train_step_counter)
tf_agent.initialize()


# In[ ]:


tf_agent.train_step_counter.assign(0)


# In[ ]:


avg_return = compute_avg_return(eval_tf_env, tf_agent.policy, num_eval_episodes)


# In[ ]:


returns = [avg_return]


# In[ ]:


returns


# In[ ]:


collect_episode(train_tf_env, tf_agent.collect_policy, py_replay_buffer, steps=2)

    
iterator = iter(py_replay_buffer.as_dataset(sample_batch_size=1))
trajectories  = next(iterator)
print(trajectories)


# In[ ]:


py_replay_buffer.clear()

for _ in range(num_iterations):
    # collect few episodes using collect_policy and save to replay buffer
    collect_episode(train_tf_env, tf_agent.collect_policy, py_replay_buffer, steps=2)
    
    iterator = iter(py_replay_buffer.as_dataset(sample_batch_size=1))
    trajectories  = next(iterator)
    print(trajectories)
    batched_exp = tf.nest.map_structure(
        lambda t: tf.expand_dims(t, axis=0),
        trajectories
    )
    
    train_loss = tf_agent.train(batched_exp).loss
    
    py_replay_buffer.clear()
    
    step = tf_agent.train_step_counter.numpy()
    
    if step % log_interval == 0:
        print('step == {0}: loss = {1}'.format(step, train_loss.loss))
        
    if step % eval_interval == 0:
        avg_return = compute_avg_return(eval_tf_env, tf_agent.policy, num_eval_episodes)
        print('step = {0}: Average return = {1}'.format(step, avg_return))
        
        returns.append(avg_return)


# In[ ]:


steps = range(0, num_iterations + 1, eval_interval)
plt.plot(steps, returns)
plt.ylabel('Average Return')
plt.xlabel('Step')
plt.ylim(top=250)


# In[ ]:




