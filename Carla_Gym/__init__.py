from gym.envs.registration import register

register(
    id='CarlaGym-v0',
    entry_point='Carla_Gym.envs:CarlaEnv',
)

