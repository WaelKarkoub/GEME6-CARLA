
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib.pyplot as plt

import gym
from gym import wrappers
import sys
import time
import pickle
sys.path.insert(0, '/home/wael/Desktop/golfcart/GEME6-CARLA/Carla_Gym/envs/')
from carla_env import CarlaEnv
use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        super(ActorCritic, self).__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
        )
        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)
        
        self.apply(init_weights)
        
    def forward(self, x):
        value = self.critic(x)
        mu    = self.actor(x)
        std   = self.log_std.exp().expand_as(mu)
        dist  = Normal(mu, std)
        return dist, value

def test_env(vis=True):
    state = env.reset()
    if vis: env.render()
    done = False
    total_reward = 0
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        dist, _ = model(state)
        next_state, reward, done, _ = env.step(dist.sample().cpu().numpy()[0])
        state = next_state
        if vis: env.render()
        total_reward += reward
    return total_reward

def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns

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
    while True:
        try:
            env = CarlaEnv()
            break
        except Exception as e:
            print(e)

    epochs = 10000
    steps  = 1800
    num_inputs  = envs.observation_space.shape[0]
    num_outputs = envs.action_space.shape[0]

    #Hyper params:
    hidden_size = 256
    lr          = 3e-2
    num_steps   = 20

    model = ActorCritic(num_inputs, num_outputs, hidden_size).to(device)
    optimizer = optim.Adam(model.parameters())

    
    replay_buffer_size = 1000000
    frame_idx   = 0
    rewards     = []
    batch_size  = 128
    while frame_idx < epochs:
        time.sleep(1/20)
        state = env.reset()
        log_probs = []
        values    = []
        rewards   = []
        masks     = []
        entropy = 0

        
        for step in range(steps):
            env.render()
            state = torch.FloatTensor(state).to(device)
            dist, value = model(state)

            action = dist.sample()
            next_state, reward, done, _ = env.step(action.cpu().numpy())
            
            log_prob = dist.log_prob(action)
            entropy += dist.entropy().mean()
            
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))
            masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))
            
            state = next_state
            frame_idx += 1
            
            if done:
                break
        
        next_state = torch.FloatTensor(next_state).to(device)
        _, next_value = model(next_state)
        returns = compute_gae(next_value, rewards, masks, values)
        
        log_probs = torch.cat(log_probs)
        returns   = torch.cat(returns).detach()
        values    = torch.cat(values)

        advantage = returns - values

        actor_loss  = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
if __name__ == "__main__":
	main()