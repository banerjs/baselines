from baselines.her.envs.oomdp_taxi import TaxiEnv
from gym import envs
from gym.envs.registration import register

if 'Taxi-OOMDP-v0' not in envs.registry.env_specs:
    register(
        id='Taxi-OOMDP-v0',
        entry_point='baselines.her.envs.oomdp_taxi:TaxiEnv',
        max_episode_steps=50,
        kwargs={
            'reward_type': 'sparse',
            'num_passengers': 1,
            'layout': 'no_walls',
        }
    )

if 'Taxi-OOMDP-v1' not in envs.registry.env_specs:
    register(
        id='Taxi-OOMDP-v1',
        entry_point='baselines.her.envs.oomdp_taxi:TaxiEnv',
        max_episode_steps=500,
        kwargs={
            'reward_type': 'sparse',
            'num_passengers': 1,
            'layout': 'diuk_7x7',
            'seed': 1,
        }
    )

if 'Taxi-OOMDP-v2' not in envs.registry.env_specs:
    register(
        id='Taxi-OOMDP-v2',
        entry_point='baselines.her.envs.oomdp_taxi:TaxiEnv',
        max_episode_steps=1000,
        kwargs={
            'reward_type': 'sparse',
            'num_passengers': 1,
            'layout': 'diuk_11x11',
            'seed': 1,
        }
    )

if 'TableSim-v0' not in envs.registry.env_specs:
    register(
        id='TableSim-v0',
        entry_point='baselines.her.envs.table_sim:TableSim',
        max_episode_steps=1000,
        kwargs={
            'reward_type': 'sparse',
            'celery_queue': 'table_sim', # Override this to instantiate multiple envs
        }
    )
