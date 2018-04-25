#!/usr/bin/env python
# This uses the David Abel's taxi domain under the hood

# Regular python imports
import sys
import os
import time
import numpy as np

# Gym environment basic imports
import gym
from gym import error, spaces, utils
from gym.utils import seeding

# Simple RL imports
from simple_rl.tasks import TaxiOOMDP

# Helper classes and functions

class TaxiLayout(object):
    """
    Different layouts of the taxi domain
    """

    def __init__(self, width, height, walls, passenger_choices=None, destination_choices=None, taxi_choices=None):
        self.width = width
        self.height = height
        self.walls = walls

        if passenger_choices is None:
            self.passenger_choices = self._free_spaces()
        else:
            self.passenger_choices = passenger_choices

        if destination_choices is None:
            self.destination_choices = self._free_spaces()
        else:
            self.destination_choices = destination_choices

        if taxi_choices is None:
            self.taxi_choices = self._free_spaces()
        else:
            self.taxi_choices = taxi_choices

    def _free_spaces(self):
        free_spaces = []
        for x in range(1, self.width+1):
            for y in range(1, self.height+1):
                loc = {"x": x, "y": y}
                if loc not in self.walls:
                    free_spaces.append(loc)
        return free_spaces

TaxiLayout.LAYOUTS = {
    'no_walls': TaxiLayout(4, 4, walls=[]),
    'diuk_7x7': TaxiLayout(
        7, 7,
        walls=[
            {"x": 1, "y": 1}, {"x": 2, "y": 1}, {"x": 3, "y": 1}, {"x": 4, "y": 1}, {"x": 5, "y": 1}, {"x": 6, "y": 1}, {"x": 7, "y": 1},
            {"x": 1, "y": 2}, {"x": 1, "y": 3}, {"x": 1, "y": 4}, {"x": 1, "y": 5}, {"x": 1, "y": 6}, {"x": 1, "y": 7},
            {"x": 7, "y": 2}, {"x": 7, "y": 3}, {"x": 7, "y": 4}, {"x": 7, "y": 5}, {"x": 7, "y": 6}, {"x": 7, "y": 7},
            {"x": 2, "y": 7}, {"x": 3, "y": 7}, {"x": 4, "y": 7}, {"x": 5, "y": 7}, {"x": 6, "y": 7},
            {"x": 3, "y": 2}, {"x": 3, "y": 3},
            {"x": 4, "y": 5}, {"x": 4, "y": 6},
            {"x": 5, "y": 2}, {"x": 5, "y": 3},
        ],
        passenger_choices=[
            {"x": 2, "y": 2}, {"x": 2, "y": 5}, {"x": 6, "y": 2}, {"x": 6, "y": 6}
        ],
        destination_choices=[
            {"x": 2, "y": 2}, {"x": 2, "y": 5}, {"x": 6, "y": 2}, {"x": 6, "y": 6}
        ]
    ),
    'diuk_11x11': TaxiLayout(
        11, 11,
        walls=[
            {"x": 1, "y": 1}, {"x": 2, "y": 1}, {"x": 3, "y": 1}, {"x": 4, "y": 1}, {"x": 5, "y": 1}, {"x": 6, "y": 1}, {"x": 7, "y": 1}, {"x": 8, "y": 1}, {"x": 9, "y": 1}, {"x": 10, "y": 1}, {"x": 11, "y": 1},
            {"x": 1, "y": 2}, {"x": 1, "y": 3}, {"x": 1, "y": 4}, {"x": 1, "y": 5}, {"x": 1, "y": 6}, {"x": 1, "y": 7}, {"x": 1, "y": 8}, {"x": 1, "y": 9}, {"x": 1, "y": 10}, {"x": 1, "y": 11},
            {"x": 11, "y": 2}, {"x": 11, "y": 3}, {"x": 11, "y": 4}, {"x": 11, "y": 5}, {"x": 11, "y": 6}, {"x": 11, "y": 7}, {"x": 11, "y": 8}, {"x": 11, "y": 9}, {"x": 11, "y": 10}, {"x": 11, "y": 11},
            {"x": 2, "y": 11}, {"x": 3, "y": 11}, {"x": 4, "y": 11}, {"x": 5, "y": 11}, {"x": 6, "y": 11}, {"x": 7, "y": 11}, {"x": 8, "y": 11}, {"x": 9, "y": 11}, {"x": 10, "y": 11},
            {"x": 3, "y": 2}, {"x": 3, "y": 3}, {"x": 3, "y": 4}, {"x": 3, "y": 5},
            {"x": 7, "y": 2}, {"x": 7, "y": 3}, {"x": 7, "y": 4}, {"x": 7, "y": 5},
            {"x": 5, "y": 7}, {"x": 5, "y": 8}, {"x": 5, "y": 9}, {"x": 5, "y": 10},
        ]
    ),
}


class MultiDiscreteHighLow(gym.Space):
    """Zero-centered multi-discrete. Incoming vectors specify the high and low
    for each dimension of the space"""
    def __init__(self, low, high):
        self.low = np.asarray(low, dtype=np.int32)
        self.high = np.asarray(high, dtype=np.int32)
        assert self.low.ndim == self.high.ndim == 1, 'low and high should be 1d arrays'
        assert self.low.shape == self.high.shape, 'low and high should have the same shape'
        gym.Space.__init__(self, (self.low.size,), np.int8)

    def sample(self):
        return np.array([
            spaces.np_random.randint(l,h) for (l,h) in zip(self.low, self.high)
        ], dtype=self.dtype)

    def contains(self, x):
        return np.logical_and(x >= self.low, x < self.high).all() and x.dtype.kind in 'ui'

    def to_jsonable(self, sample_n):
        return [sample.tolist() for sample in sample_n]

    def from_jsonable(self, sample_n):
        return np.array(sample_n)


# Create the taxi environment with the gym API that HER expects

class TaxiEnv(gym.GoalEnv):
    """
    Look at the implementation in simple_rl.tasks.TaxiOOMDP for the details
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, num_passengers=1, layout='diuk_7x7', seed=None, goal_distance_threshold=1e-7, reward_type='sparse'):
        self.seed(seed)
        self.num_passengers = num_passengers
        self.reward_type = reward_type
        self.goal_distance_threshold = goal_distance_threshold

        # Set the MDP
        self.mdp = None
        self.layout = TaxiLayout.LAYOUTS[layout]
        self.goal = self._get_goal()
        self.reset(hard=True)

        # Set the action, observation, goal spaces
        self.action_space = spaces.Box(0, 1., shape=(len(self.mdp.get_actions()),), dtype='float32')
        self.observation_space = spaces.Dict({
            'observation': MultiDiscreteHighLow(
                self.observation_limits[:,0],
                self.observation_limits[:,1]
            ),
            'desired_goal': MultiDiscreteHighLow(
                [1-self.mdp.width, 1-self.mdp.height, 0],
                [self.mdp.width, self.mdp.height, 2]
            ),
            'achieved_goal': MultiDiscreteHighLow(
                [1-self.mdp.width, 1-self.mdp.height, 0],
                [self.mdp.width, self.mdp.height, 2]
            ),
        })

    ### Private methods

    def _get_goal(self):
        # Goal representation = [
        #   pass_x-pass_dx, pass_y-pass_dy, in_taxi
        # ] for pass in range(num_passengers)

        return np.array([0., 0., 0.] * self.num_passengers)

    def _achieved_goal_from_state(self, state):
        goal = []
        for passenger in state.get_objects_of_class("passenger"):
            goal.extend([
                passenger["x"]-passenger["dest_x"],
                passenger["y"]-passenger["dest_y"],
                passenger["in_taxi"],
            ])
        return np.array(goal, dtype=np.float32)

    def _get_obs_from_state(self, state):
        return {
            "observation": state.features(),
            "achieved_goal": self._achieved_goal_from_state(state),
            "desired_goal": self.goal,
        }


    ### Env methods

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, hard=False):
        # If this is not a hard reset, then simply call the underlying reset of
        # the MDP
        if not hard:
            self.mdp.reset()
            mdp_state = self.mdp.get_curr_state()
            return self._get_obs_from_state(mdp_state)

        # Use the layout to set the taxi and passengers that need to be used to
        # set the MDP
        loc = self.np_random.choice(self.layout.taxi_choices)
        agent = {**loc, "has_passenger": 0}

        passengers = []
        for idx in range(self.num_passengers):
            loc = self.np_random.choice(self.layout.passenger_choices)
            destination = self.np_random.choice(self.layout.destination_choices)
            passengers.append({
                **loc,
                "dest_x": destination["x"], "dest_y": destination["y"],
                "in_taxi": 0
            })

        self.mdp = TaxiOOMDP(
            width=self.layout.width, height=self.layout.height,
            agent=agent, walls=self.layout.walls, passengers=passengers
        )

        # Set the limits of the observation
        mdp_state = self.mdp.get_curr_state()
        self.observation_limits = np.array([[0., 0.]] * mdp_state.get_num_feats())
        for feature, idx in mdp_state.get_feature_indices().items():
            if "has_passenger" in feature or "in_taxi" in feature:
                self.observation_limits[idx, 1] = 2.
            elif feature[-1] == "x":
                self.observation_limits[idx, 0] = 1
                self.observation_limits[idx, 1] = self.mdp.width
            elif feature[-1] == "y":
                self.observation_limits[idx, 0] = 1
                self.observation_limits[idx, 1] = self.mdp.height
            else:
                raise Exception("Unknown feature: {}".format(feature))

        return self._get_obs_from_state(mdp_state)


    def step(self, action):
        # Get the string of the action
        action = self.mdp.get_actions()[np.argmax(action)]

        reward, next_state = self.mdp.execute_agent_action(action)
        done = next_state.is_terminal()

        obs = self._get_obs_from_state(next_state)

        return (obs, reward, done, {**obs, "reward": reward})

    def compute_reward(self, achieved_goal, desired_goal, info):
        # Give the reward only if the desired goal matches the observed goal
        # d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        # if self.reward_type == 'sparse':
        #     return -(d > self.goal_distance_threshold).astype(np.float32)
        # else:
        #     return -d
        if (achieved_goal == info["achieved_goal"]).all() and (desired_goal == info["desired_goal"]).all():
            return info["reward"]
        raise Exception("Well, we're in a pickle")


    def render(self, mode="human"):
        # TODO: Since HER is not the typical agent, this needs to be changed
        pass
