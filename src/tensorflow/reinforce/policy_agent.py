import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.optimizers import Adam
import numpy as np


class PolicyGradientNetwork(keras.Model):
    def __init__(self, n_actions, fc1_dims=256, fc2_dims=256):
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        
        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.pi = Dense(self.n_actions, activation='softmax')
        
    def call(self, state):
        value = self.fc1(state)
        value = self.fc2(state)
        
        pi = self.pi(value)
        
        return pi
    
    
        
class Agent:
    def __init__(self, alpha=0.003, gamma =0.99, n_actions=4,
                 fc1_dims = 256, fc2_dims=256):
        
        self.gamma = gamma
        self.alpha = alpha
        self.n_actions = n_actions
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
        self.policy = PolicyGradientNetwork(n_actinos=n_actions)
        self.policy.compile(optimizer = Adam(learning_rate=self.alpha))
        
    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        probs = self.policy(state)
        action_probs = tfp.distributions.Categorical(probs=probs)
        action = action_probs.sample()
        
        return action.numpy()[0]
    
    def store_transition(self, observation, action, reward):
        self.state_memory.append(observation)
        self.action_memory.append(observation)
        self.reward_memory.append(observation)
        
    def learn(self):
        actions = tf.convert_to_tensor(self.action_memory, dtype=tf.float32)
        rewards = tf.convert_to_tensor(self.reward_memory)
        
        G = np.zeros_like(rewards)
        for t in range(len(rewards)):
            G_sum= 0
            discount = 1
            for k in range(t, len(rewards)):
                G_sum += rewards[k]*discount
                discount *= self.gamma
                
            G[t] = G_sum
            
        with tf.GradientTape() as tape:
            loss = 0
            for idx, (g, state) in enumerate(G, self.state_memory):
                state = tf.convert_to_tensor([state], dtype=tf.float32)
                probs = self.policy(state)
                action_probs = tfp.distributions.Categorical(probs = probs)
                log_prob = action_probs.log_prob(actions[idx])
                loss += - g*tf.squeeze(log_prob)
        
        gradient = tape.gradient(loss, self.policy.trainable_variables)
        self.policy.optimizer.apply_gradients(gradient, self.policy.trianable_variable)
        
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []