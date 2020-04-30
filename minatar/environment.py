################################################################################################################
# Authors:                                                                                                     #
# Kenny Young (kjyoung@ualberta.ca)                                                                            #
# Tian Tian (ttian@ualberta.ca)                                                                                #
################################################################################################################
from importlib import import_module
import numpy as np


#####################################################################################################################
# Environment
#
# Wrapper for all the specific game environments. Imports the environment specified by the user and then acts as a
# minimal interface. Also defines code for displaying the environment for a human user. 
#
#####################################################################################################################
class Environment:
    def __init__(self, env_name, sticky_action_prob = 0.1, difficulty_ramping = True):
        env_module = import_module('minatar.environments.'+env_name)
        self.env_name = env_name
        self.env = env_module.Env(ramping = difficulty_ramping)
        self.n_channels = self.env.state_shape()[2]
        self.sticky_action_prob = sticky_action_prob
        self.last_action = 0
        self.visualized = False
        self.closed = False

    # Wrapper for env.act
    def act(self, a):
        if(self.env._randint(0,1000)<1000.*self.sticky_action_prob):
            a = self.last_action
        self.last_action = a
        return self.env.act(a)

    # Wrapper for env.state
    def state(self):
        return self.env.state()

    # Wrapper for env.state
    def seed(self):
        return self.env.seed

    # Wrapper for env.reset
    def reset(self, seed=None):
        return self.env.reset(seed)

    # Wrapper for env.state_shape
    def state_shape(self):
        return self.env.state_shape()

    # Wrapper for env.state_shape
    def reward_range(self):
        return self.env.reward_range()

    # Wrapper for env.snapshot_size
    def snapshot_size(self):
        return self.env.snapshot_size()

    # All MinAtar environments have 6 actions
    def num_actions(self):
        return 6

    # Snapshot underlying env
    def snapshot(self):
        return self.env.snapshot()

    # Restore underlying env from snapshot
    def restore(self, snapshot):
        self.env.restore(snapshot)

    # Name of the MinAtar game associated with this environment
    def game_name(self):
        return self.env_name

    # Wrapper for env.minimal_action_set
    def minimal_action_set(self):
        return self.env.minimal_action_set()

    # Display the current environment state for time milliseconds using matplotlib
    def display_state(self, time=50):
        if(not self.visualized):
            global plt
            global colors
            global sns
            mpl = __import__('matplotlib.pyplot', globals(), locals())
            plt = mpl.pyplot
            mpl = __import__('matplotlib.colors', globals(), locals())
            colors = mpl.colors
            sns = __import__('seaborn', globals(), locals())
            self.cmap = sns.color_palette("cubehelix", self.n_channels)
            self.cmap.insert(0,(0,0,0))
            self.cmap=colors.ListedColormap(self.cmap)
            bounds = [i for i in range(self.n_channels+2)]
            self.norm = colors.BoundaryNorm(bounds, self.n_channels+1)
            _, self.ax = plt.subplots(1,1)
            plt.show(block=False)
            self.visualized = True
        if(self.closed):
            _, self.ax = plt.subplots(1,1)
            plt.show(block=False)
            self.closed = False
        state = self.env.state()
        numerical_state = np.amax(state*np.reshape(np.arange(self.n_channels)+1,(1,1,-1)),2)+0.5
        self.ax.imshow(numerical_state, cmap=self.cmap, norm=self.norm, interpolation='none')
        plt.pause(time/1000)
        plt.cla()

    def close_display(self):
        plt.close()
        self.closed = True

#####################################################################################################################
# Batch Environment
#
# Collects several environments into a batch so they can all be run in parallel.
#
#####################################################################################################################
class BatchEnvironment:
    def __init__(self, n, env_name, sticky_action_prob = 0.0, difficulty_ramping = True):
        self.n = n
        self.envs = [Environment(env_name, sticky_action_prob, difficulty_ramping) for _ in range(n)]

    # Wrapper for env.act
    def act(self, actions):
        outs = [e.act(a) for e, a in zip(self.envs, actions)]
        return tuple(np.stack(x) for x in zip(*outs))

    # Wrapper for env.state
    def state(self):
        return np.stack([e.state() for e in self.envs])

    # Wrapper for env.state
    def seeds(self):
        return np.stack([e.seed() for e in self.envs])

    # Wrapper for env.reset
    def reset(self, seeds=None):
        if seeds == None: seeds = [None] * self.n
        [e.reset(seed) for e, seed in zip(self.envs, seeds)]

    # Wrapper for env.state_shape
    def state_shape(self):
        return self.envs[0].state_shape()

    # Wrapper for env.state_shape
    def reward_range(self):
        return self.envs[0].reward_range()

    # Wrapper for env.snapshot_size
    def snapshot_size(self):
        return self.envs[0].snapshot_size()

    # All MinAtar environments have 6 actions
    def num_actions(self):
        return 6

    # Wrapper
    def snapshot(self):
        return np.stack([env.snapshot() for env in self.envs], axis=0)

    # Wrapper
    def restore(self, snapshots):
        for env, snapshot in zip(self.envs, snapshots):
            env.restore(snapshot)

    # Name of the MinAtar game associated with this environment
    def game_name(self):
        return self.envs[0].env_name

    # Wrapper for env.minimal_action_set
    def minimal_action_set(self):
        return self.envs[0].minimal_action_set()

#####################################################################################################################
# Vectorized Environment
#
# Collects several environments into a batch so they can all be run in parallel.
#
#####################################################################################################################
class VectorizedEnvironment(Environment):
    def __init__(self, n, env_name, sticky_action_prob = 0.0, difficulty_ramping = True):
        self.n = n
        env_module = import_module('minatar.environments.'+env_name)
        self.env_name = env_name
        self.env = env_module.VectorizedEnv(n, ramping = difficulty_ramping)
        self.n_channels = self.env.state_shape()[2]
        self.sticky_action_prob = sticky_action_prob
        self.last_action = 0
        if sticky_action_prob > 0.0: raise Exception("unimplemented")
        self.visualized = False
        self.closed = False

    # Wrapper for env.state
    def seeds(self):
        return self.env.seeds

    # Wrapper for env.reset
    def reset(self, n=None, seeds=None):
        return self.env.reset(n, seeds)
