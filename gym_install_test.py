import gym
import numpy as np

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate,BatchNormalization, Add
from keras.optimizers import Adam

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

import sys
import time
import random
import pickle
sys.path.insert(0, '/home/wael/Desktop/golfcart/GEME6-CARLA/Carla_Gym/envs/')
from carla_env import CarlaEnv

while True:
    try:
        env = CarlaEnv()
        break
    except Exception as e:
        print(e)

nb_actions = env.action_space.shape[0]

# Next, we build a very simple model.
actor = Sequential()
actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
actor.add(Dense(16))
# actor.add(BatchNormalization())
actor.add(Activation('relu'))
actor.add(Dense(16))
actor.add(Activation('relu'))
actor.add(Dense(16))
# actor.add(BatchNormalization())
actor.add(Activation('relu'))
actor.add(Dense(nb_actions))
actor.add(Activation('linear'))
print(actor.summary())

action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
flattened_observation = Flatten()(observation_input)

a1 = Dense(32)(action_input)
a1 = Activation('relu')(a1)
# a1 = BatchNormalization()(a1)

o1 = Dense(32)(flattened_observation)

# o1 = BatchNormalization()(o1)
o1 = Activation('relu')(o1)

# flattened_observation = Flatten()(observation_input)
x = Add()([a1, o1])
x = Dense(32)(x)
# x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dense(32)(x)
x = Activation('relu')(x)
x = Dense(32)(x)
# x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dense(1)(x)
x = Activation('linear')(x)
critic = Model(inputs=[action_input, observation_input], outputs=x)
print(critic.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=10000000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0.4, sigma=0.3)
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=50, nb_steps_warmup_actor=50,
                  random_process=random_process, gamma=.99, target_model_update=0.1)
agent.compile(Adam(lr=.0001, clipnorm=1.), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
while True:
    try:
        agent.fit(env, nb_steps=20000000, visualize=True, start_step_policy=10000,verbose=1, nb_max_episode_steps=900)
        agent.save_weights('ddpg_{}_weights.h5f'.format("Carla"), overwrite=True)
        with open("ddpg_memory2.pkl","wb") as hand:
            pickle.dump(memory,hand)
        break
    except Exception as e:
        agent.save_weights('ddpg_{}_weights.h5f'.format("Carla"), overwrite=True)
        print("model saved")
        print(e)

# After training is done, we save the final weights.
