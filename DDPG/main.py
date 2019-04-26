from ddpg import DDPG as Agent
import sys
import time
import random
import pickle
sys.path.insert(0, '/home/wael/Desktop/golfcart/GEME6-CARLA/Carla_Gym/envs/')
from carla_env import CarlaEnv
import tensorflow as tf



def main():
    with tf.Session() as sess:
        while True:
            try:
                env = CarlaEnv()
                break
            except Exception as e:
                print(e)

        agent = Agent(sess=sess,state_size=env.observation_space.shape[0], action_size=env.action_space.shape[0])
        max_episodes = 1000
        max_steps = 1800

        for i in range(int(max_episodes)):

            state = env.reset()
            print(state.shape)
            ep_reward = 0
            ep_ave_max_q = 0
            # plt.clf()
            # if i:
            #     with open("ddpg_memory.pkl","wb") as hand:
            #         pickle.dump(replay_buffer,hand)
            #     actor.save_model()
            #     critic.save_model()
            #     print("Agent saved")

            for j in range(int(max_steps)):

                print("epoch: {}, step: {}".format(i,j))
                # env.render()

                # Added exploration noise
                # a = actor.predict(np.reshape(s, (1, 3))) + (1. / (1. + i))
                action = agent.get_action(state)
                # a = controller(s[0],s[1],s[3])
                # a = [a]
                next_state, reward, done, info = env.step(action)
                print("reward: {}".format(reward))

                agent.remember(state, action, reward, done, next_state)

                # Keep adding experience to the memory until
                # there are at least minibatch size samples
                agent.train()
main()
