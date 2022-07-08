import numpy as np
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from keras.models import load_model
from tensorflow.keras.regularizers import l2
import keras.backend as K


from utils import ReplayBuffer

def masked_huber_loss(mask_value, clip_delta):
  def f(y_true, y_pred):
    error = y_true - y_pred
    cond  = K.abs(error) < clip_delta
    mask_true = K.cast(K.not_equal(y_true, mask_value), K.floatx())
    masked_squared_error = 0.5 * K.square(mask_true * (y_true - y_pred))
    linear_loss  = mask_true * (clip_delta * K.abs(error) - 0.5 * (clip_delta ** 2))
    huber_loss = tf.where(cond, masked_squared_error, linear_loss)
    return K.sum(huber_loss) / K.sum(mask_true)
  f.__name__ = 'masked_huber_loss'
  return f

class DQNModel(keras.Model):
    def __init__(self, fc1_dims:int, fc2_dims:int, n_actions: int, dueling:bool, hidden_size:int= 20, regularization_factor=0.01):
        super(DQNModel, self).__init__()
        self.dueling = dueling
        self.dense1 = keras.layers.Dense(fc1_dims, activation='relu',
                                         kernel_initializer=keras.initializers.he_normal())
        self.dense2 = keras.layers.Dense(fc2_dims, activation='relu',
                                         kernel_initializer=keras.initializers.he_normal())
        self.adv_dense = keras.layers.Dense(hidden_size, activation='relu',
                                         kernel_initializer=keras.initializers.he_normal())
        self.adv_out = keras.layers.Dense(n_actions,
                                          kernel_initializer=keras.initializers.he_normal())
        
        if dueling:
            self.v_dense = keras.layers.Dense(fc2_dims, activation='relu',kernel_initializer=keras.initializers.he_normal())
            self.v_out = keras.layers.Dense(1, kernel_initializer=keras.initializers.he_normal())
            self.lambda_layer = keras.layers.Lambda(lambda x: x - tf.reduce_mean(x))
            self.combine = keras.layers.Add()
            
    def call(self, input):
        x = self.dense1(input)
        x = self.dense2(x)
        adv = self.adv_dense(x)
        adv = self.adv_out(x)
    
        if self.dueling:
            v = self.v_dense(x)
            v = self.v_out(x)
            norm_adv = self.lambda_layer(adv)
            combined = self.combine([v, norm_adv])
            return combined
        return adv
                
                
                
         

def build_dqn(lr, n_actions, input_dims, fc1_dims, fc2_dims,regularization_factor=0.01):
    model = DQNModel(fc1_dims, fc2_dims,n_actions, dueling=True, hidden_size=32)
    # model = keras.Sequential([
        
    #     keras.layers.Dense(fc1_dims, input_shape=input_dims, activation='relu',kernel_regularizer=l2(regularization_factor)),
    #     keras.layers.Dense(fc2_dims, activation='relu',kernel_regularizer=l2(regularization_factor)),
    #     keras.layers.Dense(n_actions, activation='linear',kernel_regularizer=l2(regularization_factor))
    # ])
    #model.build(input_shape = input_dims)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    # #model.compile(optimizer=optimizer,loss=masked_huber_loss(0.0, 1.0))
    model.compile(optimizer=optimizer,loss='mean_squared_error')
    return model

class Agent():
    def __init__(self, lr, gamma, n_actions, epsilon, batch_size, input_dims, epsilon_dec=1e-4, epsilon_end = 0.01,
                 mem_size = 100000, fname='dqn_model'):
        self.action = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.model_file = fname
        self.epsilon_dec = epsilon_dec
        self.epsilon_end = epsilon_end
        self.input_dims = input_dims
        self.memory = ReplayBuffer(mem_size, input_dims)
        self.q_eval = build_dqn(lr, n_actions, input_dims, 64, 32)
        
    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)
        
    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action)
        else:
            state = np.array([observation])
            actions = self.q_eval.predict(state)
            
            action = np.argmax(actions)
            
        return action
    
    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        
        states, actions, rewards, states_, dones = self.memory.sample_buffer(self.batch_size)
        
        q_eval = self.q_eval.predict(states)
        q_next = self.q_eval.predict(states_)
        
        q_target = np.copy(q_eval)
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        #import pdb
        #pdb.set_trace()
        q_target[batch_index, actions] = rewards + self.gamma*np.max(q_next,axis=1)*dones
        
        self.q_eval.train_on_batch(states, q_target)
        self.epsilon = self.epsilon - self.epsilon_dec if self.epsilon > self.epsilon_end else self.epsilon_end
        
    
    def save_model(self):
        self.q_eval.save(self.model_file,save_format='tf')
        
    def load_model(self):
        #self.q_eval.build(self.input_dims)
        self.q_eval = keras.models.load_model(self.model_file) 
        
        
        