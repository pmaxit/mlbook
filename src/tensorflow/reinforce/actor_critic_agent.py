import os
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp


import gym
import numpy as np
    
class ActorCriticNetwork(keras.Model):
    def __init__(self, n_actions, fc1_dims=1024, fc2_dims=512, 
                 name='actor_critic', chkpt_dir='tmp/actor_critic'):
        super(ActorCriticNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir,name+'_ac')
        
        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.v = Dense(1, activation=None)
        self.pi = Dense(n_actions, activation='softmax')
        
    def call(self, state):
        value = self.fc1(state)
        value = self.fc2(value)
        
        v = self.v(value)
        pi = self.pi(value)
        
        return v, pi
    

class Agent:
    def __init__(self, alpha=0.0003, gamma=0.99, n_actions=2):
        self.gamma = gamma
        self.n_actions = n_actions
        self.action = None
                    
        self.action_space = [i for i in range(self.n_actions)]
        
        self.actor_critic = ActorCriticNetwork(n_actions=n_actions)
        
        self.actor_critic.compile(optimizer=Adam(learning_rate=alpha))    
        
    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation])
        _, probs = self.actor_critic(state)
        
        action_probabilities = tfp.distributions.Categorical(probs=probs)
        action = action_probabilities.sample()

        self.action = action
        
        return action.numpy()[0]
    
    def save_models(self):
        print('... saving models ... ')
        self.actor_critic.save_weights(self.actor_critic.checkpoint_file)
        
    def load_models(self):
        print('.... loading models ....')
        self.actor_critic.load_weights(self.actor_critic.chekcpoint_file)
        
    def learn(self, state, reward, state_, done):
        state = tf.convert_to_tensor([state])
        reward = tf.convert_to_tensor([reward])
        state_ = tf.convert_to_tensor([state_])
        done = tf.convert_to_tensor([done])
        
        with tf.GradientTape() as tape:
            state_value, probs = self.actor_critic(state)
            state_value_, _ = self.actor_critic(state_)
            
            state_value = tf.squeeze(state_value)
            state_value_ = tf.squeeze(state_value_)
            
            action_probs = tfp.distributions.Categorical(probs=probs)
            log_probs = action_probs.log_prob(self.action) # most recent action
            
            delta = reward + self.gamma*state_value_*(1 - int(done)) - state_value
            actor_loss = -log_probs*delta
            critic_loss = delta**2
            
            total_loss = actor_loss + critic_loss
            
        gradient = tape.gradient(total_loss, self.actor_critic.trainable_variables)
        self.actor_critic.optimizer.apply_gradients(zip(
            gradient, self.actor_critic.trainable_variables
        ))
        
def main():
    env = gym.make('CartPole-v0')
    agent = Agent(alpha=1e-5, n_actions=env.action_space.n)
    n_games = 1800
    
    filename = 'cartpole.png'
    figure_file = 'plots/'+filename
    
    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint=False
    
    if load_checkpoint:
        agent.load_models()
        
    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            
            if not load_checkpoint:
                agent.learn(observation, reward, observation_, done)
                
            observation = observation_
            
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        print('episode {} score {}, avg_score {}'.format(i, score, avg_score))  
        if avg_score > best_score:
            best_score = avg_score
            
            if not load_checkpoint:
                agent.save_models()
        
    #x = [i+1 for i in range(n_games)]
    #plot_learning_curve(x, score_history, figure_file)
    
if __name__ == '__main__':
    main()
