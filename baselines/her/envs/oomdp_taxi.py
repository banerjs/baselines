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

# Create the taxi environment with the gym API that HER expects

class TaxiEnv(gym.GoalEnv):
    """
    Look at the implementation in simple_rl.tasks.TaxiOOMDP for the details
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, seed=None, reward_type='sparse'):
        self.seed(seed)

    ### Env methods

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
