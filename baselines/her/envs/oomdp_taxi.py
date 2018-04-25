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
    LAYOUTS = {
        'diuk_simple': TaxiWallLayout(
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
        )
    }

    def __init__(self, width, height, walls, passenger_choices=None, destination_choices=None, taxi_choices=None):
        self.width = width
        self.height = height
        self.walls = walls

        if passenger_choices is None:
            self.passenger_choices = self._free_spaces()
        else:
            self.passenger_choices = passenger_choices

        if self.destination_choices is None:
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


# Create the taxi environment with the gym API that HER expects

class TaxiEnv(gym.GoalEnv):
    """
    Look at the implementation in simple_rl.tasks.TaxiOOMDP for the details

    - TODO: Figure out a way to randomize positions across reset calls.
    Potential issue there is making sure that the state variable positions
    correspond. Initial investigation suggests that we need to cache the index
    of each state feature and then reinitialize the state features from that
    index cache.
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, num_passengers=1, layout='diuk_simple', seed=None, reward_type='sparse'):
        self.seed(seed)
        self.num_passengers = num_passengers

        # Set the MDP
        self.mdp = None
        self.layout = TaxiLayout.LAYOUTS[layout]
        self.reset(hard=True)

        # Set the action, observation, goal spaces
        self.action_space = spaces.Discrete(len(self.mdp.get_actions()))
        # TODO: Need to figure out what the observation and goal spaces look like

    ### Env methods

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, hard=False):
        # If this is not a hard reset, then simply call the underlying reset of
        # the MDP
        if not hard:
            self.mdp.reset()
            return

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

    def step(self, action):
        # Get the string of the action
        action = self.mdp.get_actions()[action]

        reward, next_state = self.mdp.execute_agent_action(action)
        done = next_state.is_terminal()

        # TODO: Depending on the observation space definition, this needs to be set

    def compute_reward(self, achieved_goal, desired_goal, info):
        # TODO: This needs to be figured out
        pass

    def render(self, mode="human"):
        # TODO: Since HER is not the typical agent, this needs to be changed
        pass
