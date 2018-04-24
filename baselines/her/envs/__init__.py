from baselines.her.envs.oomdp_taxi import TaxiEnv
from gym.envs.registration import register

register(
    id='Taxi-OOMDP-v0',
    entry_point='baselines.her.envs.oomdp_taxi:TaxiEnv',
    reward_threshold=8,
    max_episode_steps=200,
    kwargs={
        'reward_type': 'sparse'
    }
)
