from baselines.her.envs.oomdp_taxi import TaxiEnv
from gym.envs.registration import register

register(
    id='Taxi-OOMDP-v0',
    entry_point='baselines.her.envs.oomdp_taxi:TaxiEnv',
    reward_threshold=8,
    max_episode_steps=500,
    kwargs={
        'reward_type': 'sparse',
        'num_passengers': 1,
        'layout': 'no_walls',
    }
)

register(
    id='Taxi-OOMDP-v1',
    entry_point='baselines.her.envs.oomdp_taxi:TaxiEnv',
    reward_threshold=8,
    max_episode_steps=500,
    kwargs={
        'reward_type': 'sparse',
        'num_passengers': 1,
        'layout': 'diuk_7x7',
        'seed': 1,
    }
)
