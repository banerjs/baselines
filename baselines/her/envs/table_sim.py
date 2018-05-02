#!/usr/bin/env python
# The Python 3 interface to OpenAI gym for the task simulator

# Regular python imports
import sys
import os
import time
import pickle
import numpy as np

# Gym environment basic imports
import gym
from gym import error, spaces, utils
from gym.utils import seeding

# Get the table_sim executor
from .celery_executor import setup_table_sim, app as CeleryApp

# Tensorflow
import tensorflow as tf

# Create the table_sim environment with the gym API that HER expects

class TableSim(gym.GoalEnv):
    """
    Use celery to communicate with the ROS table simulator
    """

    metadata = {'render.modes': ['human']}

    def __init__(
        self, seed=None, reward_type='sparse', goal_distance_threshold=1e-7,
        actions_filename='/home/banerjs/Workspaces/baselines/src/task_sim/data/task4/metadata/A_keys.pkl',
        relation_keys_filename='/home/banerjs/Workspaces/baselines/src/task_sim/data/task4/metadata/R_keys.pkl',
        result_timeout = 10, celery_queue='celery',
        *args, **kwargs
    ):
        self.reward_type = reward_type
        self.goal_distance_threshold = goal_distance_threshold

        self.sim = setup_table_sim(False)
        self.execute_sim = self.sim['execute'].apply_async
        self.reset_sim = self.sim['reset'].apply_async
        self.query_state = self.sim['query_state'].apply_async
        self.celery_queue = celery_queue
        self.result_timeout = result_timeout

        with open(actions_filename, 'rb') as fd:
            self.actions_list = pickle.load(fd)

        with open(relation_keys_filename, 'rb') as fd:
            self.relations_list = pickle.load(fd)

        self.relations_dict = {x: i for i,x in enumerate(self.relations_list)}

        # Seed
        self.seed(seed)

        # Set the action space and the observation space
        self.goal = self._get_goal()
        self.action_space = spaces.Box(
            0., 1., shape=(len(self.actions_list),), dtype='float32'
        )
        self.observation_space = spaces.Dict({
            'observation': spaces.MultiDiscrete(2. * np.ones((len(self.relations_list),))),
            'desired_goal': spaces.MultiDiscrete(2. * np.ones((len(self.goal),))),
            'achieved_goal': spaces.MultiDiscrete(2. * np.ones((len(self.goal),))),
        })

        # Check that the celery queue is running
        self._check_celery_queue()


    # Private methods

    def _check_celery_queue(self):
        I = CeleryApp.control.inspect()
        active_queues = I.active_queues()

        for hostname, queues in active_queues.items():
            if self.celery_queue == hostname.split('@')[1]:
                return True

        raise error.Error("Celery queue, {}, does not exist. Checked: {}".format(
            self.celery_queue,
            active_queues.keys()
        ))

    def _get_goal(self):
        # NOTE: This is at the moment only applicable for the case of the one
        # object, one drawer case. Need to make it more extensible
        #
        # Possible goals:  'apple_inside_drawer': True, 'apple_on_drawer': True,
        # 'apple_on_stack': True, 'apple_touching_drawer': True,
        # 'apple_touching_stack': True, 'drawer_closing_stack': True

        return np.array([1, 1, 1, 1], dtype=np.int8) # [inside, closing, touching_s, touching_d]

    def _achieved_goal_from_state(self, state):
        goal = np.array([
            state[self.relations_dict["apple_inside_drawer"]],
            state[self.relations_dict["drawer_closing_stack"]],
            state[self.relations_dict["apple_touching_stack"]],
            state[self.relations_dict["apple_touching_drawer"]],
        ], dtype=np.int8)
        return goal

    def _get_obs_from_state(self, state):
        return {
            "observation": np.array(state, dtype=np.int8),
            "achieved_goal": self._achieved_goal_from_state(state),
            "desired_goal": self.goal
        }

    # Env methods

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        result = self.reset_sim(queue=self.celery_queue)
        if not result.get(timeout=self.result_timeout):
            raise error.Error("Reset sim has failed")

        result = self.query_state(queue=self.celery_queue)
        state = result.get(timeout=self.result_timeout)
        if not state or len(state) != len(self.relations_list):
            raise error.Error("Failed to get state. Returned: {}".format(state))

        return self._get_obs_from_state(state)

    def step(self, actions):
        # import json
        # print(json.dumps({ 'actions': actions.tolist() }))

        result = self.execute_sim(
            ({ 'actions': actions.tolist() },),
            queue=self.celery_queue
        )
        state = result.get(timeout=self.result_timeout)
        if not state or len(state) != len(self.relations_list):
            raise error.Error("Failed to execute. Returned: {}".format(state))

        obs = self._get_obs_from_state(state)
        reward = self.compute_reward(obs["achieved_goal"], self.goal, {})
        done = True if reward > 0. else False # NOTE: Perhaps this should be smarter?

        return (obs, reward, done, {}) # NOTE: Perhaps we should add more info?

    def compute_reward(self, achieved_goal, desired_goal, info):
        d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        if self.reward_type == 'sparse':
            return -(d > self.goal_distance_threshold).astype(np.float32)
        else:
            return -d

    def render(self, mode="human"):
        # TODO: Need to configure celery without its stupid logging
        pass
