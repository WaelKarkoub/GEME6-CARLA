import gym

import sys
import time
import random
import pickle
sys.path.insert(0, '/home/wael/Desktop/golfcart/GEME6-CARLA/Carla_Gym/envs/')
from carla_env import CarlaEnv
import numpy as np

import matplotlib.pyplot as plt
from controller import Controller

# env = CarlaEnv()
while True:
    try:
        env = CarlaEnv()
        break
    except Exception as e:
        print(e)
max_episodes = 100
max_steps = 1800
control = Controller()


for i in range(int(max_episodes)):

    s = env.reset()
    
    counter = 0
    while True:
        if counter == 0:
            a = control.action(s[0],s[3],s[1],0.001,controller_type="LQR")
        else:
            a = control.action(s[0],s[3],s[1],s[2],controller_type="LQR")

        s2, r, terminal, info = env.step(a[0])

        s = s2
        counter += 1
        if terminal:
            history_pos = last_info["history"]
            history_xte = last_info["xte_history"]
            history_vel = last_info["velError_history"]
            with open("stored_data/car_history_lqr_2.pkl","wb") as hand:
                pickle.dump(history_pos,hand)
            with open("stored_data/car_history_lqr2_xte.pkl",'wb') as hand:
                pickle.dump(history_xte,hand)
            with open("stored_data/car_history_lqr2_vel.pkl",'wb') as hand:
                pickle.dump(history_vel,hand)
            break
        last_info = info
        # plt.scatter(j,r)
        # plt.xlabel("step")
        # plt.ylabel("Reward")
        # plt.draw()