# Create the OOMDP taxi domain and configure it to be able to run with HER
# NOTE: Prefer using david-abel's taxi domain rather than this hacked together
# domain

import sys
import os
import numpy as np

from six import StringIO

import gym
from gym import error, spaces, utils
from gym.utils import seeding

# Auxiliary OOMDP classes

# Items in the world that have attributes
class Item(object):
    """All items have an x,y,name,unique_name and the relations that they
    belong to"""

    # X,Y for all items will be in the specified range, end-points inclusive.
    # Can change this value for all items individually if needed
    grid_bounds = np.array([[-0.5, 4.5], [-0.5, 4.5]])

    def __init__(self, x, y, name, unique_name, relations=set()):
        self.x = x
        self.y = y
        self.name = name
        self.unique_name = unique_name
        self.relations = relations
        self.modified = False # Flag to keep track of if the object has been modified

    # Helpers to calculate the relations
    def touchN(self, item):
        return (self.x > item.x >= self.x-1) and (self.y == item.y)

    def touchS(self, item):
        return (self.x < item.x <= self.x+1) and (self.y == item.y)

    def touchE(self, item):
        return (self.x == item.x) and (self.y < item.y <= self.y+1)

    def touchW(self, item):
        return (self.x == item.x) and (self.y > item.y >= self.y-1)

    def on(self, item):
        return (self.x == item.x) and (self.y == item.y)


class Taxi(Item):
    """The taxi in the domain"""
    def __init__(self, x, y):
        super().__init__(x, y, 'taxi', 'taxi')

class Passenger(Item):
    """A passenger in the domain. We allow multiple passengers and the desired
    goal of the passenger is part of the attributes"""
    _num_passengers = -1 # Static to keep track of the number of passengers

    def __init__(self, x, y, dest, taxi):
        Passenger._num_passengers += 1
        super().__init__(
            x, y, 'pass', 'pass{}'.format(Passenger._num_passengers)
        )

        self.in_taxi = (x == taxi.x and y == taxi.y)
        self.dest_x = dest.x
        self.dest_y = dest.y

class Destination(Item):
    """A location in the grid where the passenger can have a desire to get to"""
    _num_destinations = -1 # Static to keep track of the number of destinations

    def __init__(self, x, y):
        Destination._num_destinations += 1
        super().__init__(
            x, y, 'dest', 'dest{}'.format(Destination._num_destinations)
        )

class Wall(Item):
    """An obstruction in the world"""
    _num_walls = -1 # Static to keep track of the number of walls

    def __init__(self, x, y):
        Wall._num_walls += 1
        super().__init__(x, y, 'wall', 'wall{}'.format(Wall._num_walls))


# Relations in the OOMDP world
class Relation(object):
    """Defines a relation between two objects"""
    RELATION_TYPES = ['touchN', 'touchS', 'touchE', 'touchW', 'on']

    def __init__(self, obj1, obj2, kind):
        assert(kind in Relation.RELATION_TYPES)

        self.kind = kind
        self.obj1 = obj1
        self.obj2 = obj2

        name_template = "{0}_{1}_{2}"
        self.name = name_template.format(obj1.name, kind, obj2.name)
        self.unique_name = name_template.format(obj1.unique_name, kind, obj2.unique_name)

        self._value = None

    def __unicode__(self):
        return "{}({}, {}): {}".format(
            self.kind,
            self.obj1.unique_name,
            self.obj2.unique_name,
            self.value
        )

    def __repr__(self):
        return self.__unicode__()

    def __hash__(self):
        return (hash(self.kind) ^ hash(self.obj1.unique_name) ^ hash(self.obj2.unique_name))

    @property
    def value(self):
        self._recalculate_value()
        return self._value

    def _recalculate_value(self, force=False):
        if force or self.obj1.modified or self.obj2.modified or self._value is None:
            self._value = getattr(self.obj1, self.kind)(self.obj2)



# The main gym env class

MAP = [
    "+---------+",
    "| : | : : |",
    "| : | : : |",
    "| : : : : |",
    "| | : | : |",
    "| | : | : |",
    "+---------+",
]

class TaxiEnv(gym.GoalEnv):
    """
    The Taxi Problem, but with the state space defined as in Diuk et al. - the
    OOMDP representation
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(
        self,
        difficulty=1, # Standard 5x5 taxi domain
        bounds=[[-0.5, 4.5], [-0.5, 4.5]],
        seed=0,
        reward_type='sparse'
    ):
        if difficulty != 1:
            raise error.Error("Unknown difficulty level {}".format(difficulty))

        self.difficulty = difficulty

        self.seed(seed)
        self.bounds = np.array(bounds)
        self.lastaction = None
        self.relations = None

        self.state_feature_keys = {}

        # Reset the internal counters for the classes
        Passenger._num_passengers = -1
        Destination._num_destinations = -1
        Wall._num_walls = -1

        # Set the map
        self.desc = np.asarray(MAP,dtype='c')

        # Set the action space
        self.action_space = spaces.Discrete(6)

        # Reset the state
        self.world_state = None
        self.reset(True)

    # Env methods

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, new_world_state=False):
        # For difficulty of 1: grid of 5x5 and 1 passenger and 1 destination

        # Create the taxi
        taxi_pos = (
            self.np_random.randint(*(self.bounds[0,:]+0.5)),
            self.np_random.randint(*(self.bounds[1,:]+0.5))
        )
        self.taxi = Taxi(*taxi_pos)

        # Create the passenger and destinations
        self.acceptable_destinations = np.array([[0,0], [0,4], [4,0], [4,3]])
        passenger_pos = self.acceptable_destinations[
            self.np_random.choice(len(self.acceptable_destinations), 1)
        ]
        destination_pos = self.acceptable_destinations[
            self.np_random.choice(len(self.acceptable_destinations), 1)
        ]
        self.destinations = [
            Destination(*pos) for pos in destination_pos
        ]
        self.passengers = [
            Passenger(*pos, self.destinations[idx], self.taxi)
            for idx, pos in enumerate(passenger_pos)
        ]

        # Create the walls
        wall_positions = [
            (-0.5, 0), (-0.5, 1), (-0.5, 2), (-0.5, 3), (-0.5, 4),
            (0, -0.5), (1, -0.5), (2, -0.5), (3, -0.5), (4, -0.5),
            (0, 4.5),  (1, 4.5),  (2, 4.5),  (3, 4.5),  (4, 4.5),
            (4.5, 0),  (4.5, 1),  (4.5, 2),  (4.5, 3),  (4.5, 4),
            (0, 1.5),  (1, 1.5),  (3, 0.5),  (4, 0.5),  (3, 2.5),  (4, 2.5),
        ]
        self.walls = [
            Wall(*pos) for pos in wall_positions
        ]

        # Create the relations
        self.relations = {}

        # First the taxi
        for item in (self.passengers + self.destinations + self.walls):
            for kind in Relation.RELATION_TYPES:
                relation = Relation(self.taxi, item, kind)
                self.relations[relation.unique_name] = relation
                self.taxi.relations.add(relation)

        # Next the passengers
        for idx, passenger in enumerate(self.passengers):
            for item in (self.passengers[idx+1:] + self.destinations + self.walls):
                for kind in Relation.RELATION_TYPES:
                    relation = Relation(passenger, item, kind)
                    self.relations[relation.unique_name] = relation
                    passenger.relations.add(relation)

        # Next the destinations
        for idx, destination in enumerate(self.destinations):
            for item in (self.destinations[idx+1:] + self.walls):
                for kind in Relation.RELATION_TYPES:
                    relation = Relation(destination, item, kind)
                    self.relations[relation.unique_name] = relation
                    destination.relations.add(relation)

        # Finally the walls (for confusion)
        # for idx, wall in enumerate(self.walls):
        #     for item in self.walls[idx+1:]:
        #         for kind in Relation.RELATION_TYPES:
        #             relation = Relation(wall, item, kind)
        #             self.relations[relation.unique_name] = relation
        #             wall.relations.add(relation)

        # Create the observation space, if it hasn't been already, otherwise
        # set the indices.
        if self.observation_space is None or new_world_state:
            world_state = []
            world_state_bounds = []

        if self.observation_space is None or new_world_state:
            world_state.extend([self.taxi.x, self.taxi.y])
            world_state_bounds.extend(self.bounds[:,1] + 0.5)
            self.state_feature_keys[self.taxi.unique_name+'_x'] = 0
            self.state_feature_keys[self.taxi.unique_name+'_y'] = 1
        else:
            self.world_state[0] = self.taxi.x
            self.world_state[1] = self.taxi.y

        idx = 2
        for passenger in self.passengers:
            if self.observation_space is None or new_world_state:
                world_state.extend([
                    passenger.x, passenger.y,
                    int(passenger.in_taxi),
                    passenger.dest_x, passenger.dest_y
                ])
                world_state_bounds.extend(
                    list(self.bounds[:,1]+0.5) + [2] + list(self.bounds[:,1]+0.5)
                )
                self.state_feature_keys[passenger.unique_name+'_x'] = idx
                self.state_feature_keys[passenger.unique_name+'_y'] = idx+1
                self.state_feature_keys[passenger.unique_name+'_it'] = idx+2 # in_taxi
                self.state_feature_keys[passenger.unique_name+'_dx'] = idx+3
                self.state_feature_keys[passenger.unique_name+'_dy'] = idx+4
            else:
                self.world_state[idx:idx+5] = np.array([
                    passenger.x, passenger.y, passenger.in_taxi,
                    passenger.dest_x, passenger.dest_y
                ], dtype=np.int)
            idx += 5

        for destination in self.destinations:
            if self.observation_space is None or new_world_state:
                world_state.extend([destination.x, destination.y])
                world_state_bounds.extend(self.bounds[:,1]+0.5)
                self.state_feature_keys[destination.unique_name+'_x'] = idx
                self.state_feature_keys[destination.unique_name+'_y'] = idx+1
            else:
                self.world_state[idx:idx+2] = np.array([destination.x, destination.y])
            idx += 2

        for wall in self.walls:
            pos = [
                int(wall.x if isinstance(wall.x, int) else wall.x+0.5),
                int(wall.y if isinstance(wall.y, int) else wall.y+0.5)
            ]
            if self.observation_space is None or new_world_state:
                world_state.extend(pos)
                world_state_bounds.extend(self.bounds[:,1]+1.5)
                self.state_feature_keys[wall.unique_name+'_x'] = idx
                self.state_feature_keys[wall.unique_name+'_y'] = idx+1
            else:
                self.world_state[idx:idx+2] = pos
            idx += 2

        for relation_key in sorted(self.relations.keys()):
            if self.observation_space is None or new_world_state:
                world_state.append(int(self.relations[relation_key].value))
                world_state_bounds.append(2)
                self.state_feature_keys[relation_key] = idx
            else:
                self.world_state[idx] = self.relations[relation_key].value
            idx += 1


        # Set the space, if it hasn't been already
        # TODO: Other possible goals: Same x/y for passenger and dest...
        if self.observation_space is None or new_world_state:
            self.world_state = np.array(world_state, dtype=np.int)
            world_state_bounds = np.array(world_state_bounds, dtype=np.int)
            self.observation_space = spaces.Dict({
                'observation': spaces.MultiDiscrete(world_state_bounds),
                'desired_goal': spaces.MultiDiscrete(
                    world_state_bounds[[
                        self.state_feature_keys['pass0_it'],
                        self.state_feature_keys['pass0_on_dest0']
                    ]]
                ),
                'achieved_goal': spaces.MultiDiscrete(
                    world_state_bounds[[
                        self.state_feature_keys['pass0_it'],
                        self.state_feature_keys['pass0_on_dest0']
                    ]]
                ),
            })

        # Set the goal regardless
        self.goal = np.array([0, 1]) # pass0_it, pass0_on_dest0

        # Return the first observation
        return {
            'observation': self.world_state.copy(),
            'achieved_goal': self.world_state[[
                        self.state_feature_keys['pass0_it'],
                        self.state_feature_keys['pass0_on_dest0']
                    ]].copy(),
            'desired_goal': self.goal.copy()
        }


    def render(self, mode="human"):
        # Copy parts from the original gym taxi. Make the assumption that the
        # walls don't move and therefore do not need to be re-rendered
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        out = self.desc.copy().tolist()
        out = np.array([[c.decode('utf-8') for c in line] for line in out])

        def pos(item): return int(item.x+1), int((item.y+0.5)*2)

        if self.passengers[0].in_taxi:
            out[pos(self.taxi)] = "_"
        else:
            out[pos(self.taxi)] = "^"
            out[pos(self.passengers[0])] = "P"

        out[pos(self.destinations[0])] = "D"

        outfile.write("\n".join(["".join(row) for row in out])+"\n")
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["South", "North", "East", "West", "Pickup", "Dropoff"][self.lastaction]))
        else: outfile.write("\n")

        # No need to return anything for human
        if mode != 'human':
            return outfile

    def step(self, a):
        reward = -1
        self.lastaction = a

        modified_items = set()

        # First see if we can figure out the new state observation
        if a == 0: # South
            if self.taxi.x+1 <= self.bounds[0,1]-0.5:
                self.taxi.x += 1
                self.taxi.modified = True
                modified_items.add(self.taxi)

                for passenger in self.passengers:
                    if passenger.in_taxi:
                        passenger.x = self.taxi.x
                        passenger.modified = True
                        modified_items.add(passenger)

        elif a == 1: # North
            if self.taxi.x-1 >= self.bounds[0,0]+0.5:
                self.taxi.x -= 1
                self.taxi.modified = True
                modified_items.add(self.taxi)

                for passenger in self.passengers:
                    if passenger.in_taxi:
                        passenger.x = self.taxi.x
                        passenger.modified = True
                        modified_items.add(passenger)

        elif a == 2: # East
            if self.taxi.y+1 <= self.bounds[1,1]-0.5 \
            and self.desc[self.taxi.x+1, 2*self.taxi.y+2] == b":":
                self.taxi.y += 1
                self.taxi.modified = True
                modified_items.add(self.taxi)

                for passenger in self.passengers:
                    if passenger.in_taxi:
                        passenger.y = self.taxi.y
                        passenger.modified = True
                        modified_items.add(passenger)

        elif a == 3: # West
            if self.taxi.y-1 >= self.bounds[1,0]+0.5 \
            and self.desc[self.taxi.x+1, 2*self.taxi.y] == b":":
                self.taxi.y -= 1
                self.taxi.modified = True
                modified_items.add(self.taxi)

                for passenger in self.passengers:
                    if passenger.in_taxi:
                        passenger.y = self.taxi.y
                        passenger.modified = True
                        modified_items.add(passenger)

        elif a == 4: # Pickup
            picked_up = False
            for idx, passenger in enumerate(self.passengers):
                if self.taxi.x == passenger.x and self.taxi.y == passenger.y and not passenger.in_taxi:
                    picked_up = True
                    passenger.in_taxi = True
                    passenger.modified = True
                    modified_items.add(passenger)

            if not picked_up:
                reward = -10

        elif a == 5: # Dropoff
            if (self.taxi.x, self.taxi.y) in self.acceptable_destinations:
                dropped_off = False
                for passenger in self.passengers:
                    if passenger.in_taxi:
                        dropped_off = True
                        passenger.in_taxi = False
                        passenger.modified = True
                        modified_items.add(passenger)

                        if passenger.x == passenger.dest_x and passenger.y == passenger.dest_y:
                            reward = 20

                if not dropped_off:
                    reward = -10
            else:
                reward = -10

        # Based on the modifications, update the world state
        for item in modified_items:
            for relation in item.relations:
                self.world_state[self.state_feature_keys[relation.unique_name]] = relation.value

            self.world_state[self.state_feature_keys[item.unique_name+'_x']] = item.x
            self.world_state[self.state_feature_keys[item.unique_name+'_y']] = item.y

            if item.name == 'pass':
                self.world_state[self.state_feature_keys[item.unique_name+'_it']] = item.in_taxi
                self.world_state[self.state_feature_keys[item.unique_name+'_dx']] = item.dest_x
                self.world_state[self.state_feature_keys[item.unique_name+'_dy']] = item.dest_y

            item.modified = False

        # Now check the goal completion
        done = True
        for passenger in self.passengers:
            done = done and (passenger.x == passenger.dest_x) \
                and (passenger.y == passenger.dest_y) and not passenger.in_taxi


        # Send the observation and the info
        obs = {
            'observation': self.world_state.copy(),
            'achieved_goal': self.world_state[[
                        self.state_feature_keys['pass0_it'],
                        self.state_feature_keys['pass0_on_dest0']
                    ]].copy(),
            'desired_goal': self.goal.copy(),
        }
        info = {
            'world_state': self.world_state.copy(),
            'reward': reward,
        }

        return (obs, reward, done, info)

    def compute_reward(self, achieved_goal, desired_goal, info):
        # We've already computed the reward before this, don't bother
        # recalculating
        if np.any(achieved_goal != desired_goal):
            return info['reward']
        else:
            return 0
