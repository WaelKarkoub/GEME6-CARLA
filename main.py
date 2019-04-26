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


while True:
    try:
        env = CarlaEnv()
        break
    except Exception as e:
        print(e)
max_episodes = 100
control = Controller()


for i in range(int(max_episodes)):

    s = env.reset()

    counter = 0
    while True:
        if counter == 0:
            a = control.action(s[0],s[3],s[1],0.001,controller_type="LQR")
        else:
            a = control.action(s[0],s[3],s[1],s[2],controller_type="LQR")
            print(a)
        s2, r, terminal, info = env.step(a[0])

        s = s2
        counter += 1
        if terminal:
            break

        # plt.scatter(j,r)
        # plt.xlabel("step")
        # plt.ylabel("Reward")
        # plt.draw()