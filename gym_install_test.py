import gym
from gym import wrappers

import sys
import time
import random
import pickle
sys.path.insert(0, '/home/wael/Desktop/golfcart/GEME6-CARLA/Carla_Gym/envs/')
from carla_env import CarlaEnv
import numpy as np
import keras.backend as K
import tensorflow as tf

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate, Dropout
from keras.optimizers import Adam
from keras.layers.merge import Add, Multiply, Concatenate


from collections import deque
#https://github.com/spiglerg/DQN_DDQN_Dueling_and_DDPG_Tensorflow/blob/master/modules/ddpg.py
#https://towardsdatascience.com/reinforcement-learning-w-keras-openai-actor-critic-models-f084612cfd69
class ActorCritic:
    def __init__(self, env, sess):
        self.env  = env
        self.sess = sess

        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = .995
        self.gamma = .95
        self.tau   = .125

        # ===================================================================== #
        #                               Actor Model                             #
        # Chain rule: find the gradient of chaging the actor network params in  #
        # getting closest to the final value network predictions, i.e. de/dA    #
        # Calculate de/dA as = de/dC * dC/dA, where e is error, C critic, A act #
        # ===================================================================== #

        self.memory = deque(maxlen=2000000)
        self.actor_state_input, self.actor_model = self.create_actor_model()
        _, self.target_actor_model = self.create_actor_model()

        self.actor_critic_grad = tf.placeholder(tf.float32, 
            [None, self.env.action_space.shape[0]]) # where we will feed de/dC (from critic)
        
        actor_model_weights = self.actor_model.trainable_weights
        self.actor_grads = tf.gradients(self.actor_model.output, 
            actor_model_weights, -self.actor_critic_grad) # dC/dA (from actor)
        grads = zip(self.actor_grads, actor_model_weights)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)

        # ===================================================================== #
        #                              Critic Model                             #
        # ===================================================================== #		

        self.critic_state_input, self.critic_action_input, \
            self.critic_model = self.create_critic_model()
        _, _, self.target_critic_model = self.create_critic_model()

        self.critic_grads = tf.gradients(self.critic_model.output, 
            self.critic_action_input) # where we calcaulte de/dC for feeding above
        
        # Initialize for later gradient calculations
        self.sess.run(tf.initialize_all_variables())

    # ========================================================================= #
    #                              Model Definitions                            #
    # ========================================================================= #

    def create_actor_model(self):
        state_input = Input(shape=self.env.observation_space.shape)
        h1 = Dense(128, activation='relu')(state_input)
        h2 = Dense(256, activation='relu')(h1)
        h3 = Dense(128, activation='relu')(h2)
        h4 = Dense(64, activation='relu')(h3)
        output = Dense(self.env.action_space.shape[0])(h4)
        
        model = Model(input=state_input, output=output)
        adam  = Adam(lr=0.003)
        model.compile(loss="mse", optimizer=adam)
        return state_input, model

    def create_critic_model(self):
        state_input = Input(shape=self.env.observation_space.shape)
        state_h1 = Dense(256, activation='relu')(state_input)
        state_h2 = Dense(128)(state_h1)
        
        action_input = Input(shape=self.env.action_space.shape)
        action_h1    = Dense(128)(action_input)
        
        merged    = Add()([state_h2, action_h1])
        merged_h1 = Dense(64, activation='relu')(merged)
        output = Dense(1)(merged_h1)
        model  = Model(input=[state_input,action_input], output=output)
        
        adam  = Adam(lr=0.003)
        model.compile(loss="mse", optimizer=adam)
        return state_input, action_input, model

    # ========================================================================= #
    #                               Model Training                              #
    # ========================================================================= #

    def remember(self, cur_state, action, reward, new_state, done):
        self.memory.append([cur_state, action, reward, new_state, done])

    def _train_actor(self, samples):
        for sample in samples:
            cur_state, action, reward, new_state, _ = sample
            predicted_action = self.actor_model.predict(cur_state)
            grads = self.sess.run(self.critic_grads, feed_dict={
                self.critic_state_input:  cur_state,
                self.critic_action_input: predicted_action
            })[0]

            self.sess.run(self.optimize, feed_dict={
                self.actor_state_input: cur_state,
                self.actor_critic_grad: grads
            })
            
    def _train_critic(self, samples):
        for sample in samples:
            cur_state, action, reward, new_state, done = sample
            if not done:
                target_action = self.target_actor_model.predict(new_state)
                future_reward = self.target_critic_model.predict(
                    [new_state, target_action])[0][0]
                reward += self.gamma * future_reward
            self.critic_model.fit([cur_state, action], reward, verbose=0)
        
    def train(self):
        batch_size = 128
        if len(self.memory) < batch_size:
            return

        rewards = []
        samples = random.sample(self.memory, batch_size)
        self._train_critic(samples)
        self._train_actor(samples)
        self.update_target()

    # ========================================================================= #
    #                         Target Model Updating                             #
    # ========================================================================= #

    def _update_actor_target(self):
        actor_model_weights  = self.actor_model.get_weights()
        actor_target_weights = self.target_actor_model.get_weights()
        
        for i in range(len(actor_target_weights)):
            actor_target_weights[i] = actor_model_weights[i]
        self.target_actor_model.set_weights(actor_target_weights)

    def _update_critic_target(self):
        critic_model_weights  = self.critic_model.get_weights()
        critic_target_weights = self.target_critic_model.get_weights()
        
        for i in range(len(critic_target_weights)):
            critic_target_weights[i] = critic_model_weights[i]
        self.target_critic_model.set_weights(critic_target_weights)		

    def update_target(self):
        self._update_actor_target()
        self._update_critic_target()

    # ========================================================================= #
    #                              Model Predictions                            #
    # ========================================================================= #

    def act(self, cur_state):
        self.epsilon *= self.epsilon_decay
        # if np.random.random() < self.epsilon:
        #     print("Random")
        #     return self.env.action_space.sample()
        return self.actor_model.predict(cur_state)

    def save_model(self):
        with open("Actor_Agent.model","wb") as hand:
            pickle.dump(self.actor_model,hand)
        with open("Critic_Agent.model","wb") as hand:
            pickle.dump(self.critic_model,hand)
        
        with open("Agent_memory.pickle","wb") as hand:
            pickle.dump(self.memory,hand)

    def load_model(self):
        with open("Actor_Agent.model","rb") as hand:
            self.actor_model = pickle.load(hand)
        with open("Critic_Agent.model","rb") as hand:
            self.critic_model = pickle.load(hand)
        with open("Agent_memory.pickle","rb") as hand:
            self.memory = pickle.load(hand)
            print("Length: {}".format(len(self.memory)))

        self.update_target()
        
        # self.actor_grads = tf.gradients(self.actor_model.output, 
        #     actor_model_weights, -self.actor_critic_grad) # dC/dA (from actor)
        # grads = zip(self.actor_grads, actor_model_weights)
        # self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)

        # self.critic_grads = tf.gradients(self.critic_model.output, self.critic_action_input) # where we calcaulte de/dC for feeding above
        
        # # Initialize for later gradient calculations
        # self.sess.run(tf.initialize_all_variables())
def controller(xte,velError,angle):

    if angle > 50.0:
        angle = 50.0
    
    if angle < -50.0:
        angle = -50.0

    steering = angle*1/50.0

    if velError > 2:
        velError = 2
    if velError < -2:
        velError = -2

    prop = -(1.0/2.0)*velError

    if prop < 0:
        brake = np.abs(prop)
        gas = 0
    elif prop > 0:
        gas = np.abs(prop)
        brake = 0
    else:
        gas, brake = 0, 0

    return np.array([prop,steering])

def main():
    sess = tf.Session()
    K.set_session(sess)
    while True:
        try:
            env = CarlaEnv()
            break
        except Exception as e:
            print(e)
    actor_critic = ActorCritic(env, sess)
    try:
        actor_critic.load_model()
        print("\033[93m model loaded")
        
    except Exception as e:
        print(e)
    num_trials = 10000
    trial_len  = 1800

    for epochs in range(num_trials):
        while True:
            try:
                time.sleep(1/20)
                cur_state = env.reset()
                break
            except Exception as e:
                print(e)
        counter = 0

        while True:
            env.render()
            cur_state = cur_state.reshape((1, env.observation_space.shape[0]))
            # if epochs < 100:
            #     action = env.action_space.sample()
            # else:
            #     action = actor_critic.act(cur_state)
            
            action = actor_critic.act(cur_state)
            # action = controller(cur_state[0][0],cur_state[0][1],cur_state[0][2])
            action = action.reshape((1, env.action_space.shape[0]))

            new_state, reward, done, _ = env.step(action)
            
            print("reward: {}, iter: {}, epoch: {}".format(reward, counter,epochs))
            new_state = new_state.reshape((1, env.observation_space.shape[0]))
            actor_critic.remember(cur_state, action, [reward], new_state, done)
            # if (counter % 5 == 0):
            #     while True:
            #         try:
            #             print("training")
            #             actor_critic.train()
            #             break
            #         except Exception as e:
            #             print(e)
            cur_state = new_state
            if done or (counter>trial_len):
                actor_critic.save_model()
                break
            counter += 1
            if (counter % 300 == 0):
                actor_critic.save_model()
                print("Model Saved")
            
if __name__ == "__main__":
	main()