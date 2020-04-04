################################################################################################################
# Authors:                                                                                                     #
# Kenny Young (kjyoung@ualberta.ca)                                                                            #
# Tian Tian (ttian@ualberta.ca)                                                                                #
################################################################################################################
import numpy as np
from ..pseudorandom import seeded_randint

#####################################################################################################################
# Constants
#
#####################################################################################################################
max_clock = 2500

#####################################################################################################################
# Env
#
# The player controls a paddle on the bottom of the screen and must bounce a ball tobreak 3 rows of bricks along the 
# top of the screen. A reward of +1 is given for each brick broken by the ball.  When all bricks are cleared another 3 
# rows are added. The ball travels only along diagonals, when it hits the paddle it is bounced either to the left or 
# right depending on the side of the paddle hit, when it hits a wall or brick it is reflected. Termination occurs when
# the ball hits the bottom of the screen. The balls direction is indicated by a trail channel.
#
#####################################################################################################################
class Env:
    def __init__(self, ramping = None):
        self.channels ={
            'paddle':0,
            'ball':1,
            'trail':2,
            'brick':3,
        }
        self.action_map = ['n','l','u','r','d','f']
        self.inverse_action_map = {self.action_map[i]: i for i in range(len(self.action_map))}
        self.reset()

    # Update environment according to agent action
    def act(self, a):
        r = 0
        if(self.terminal):
            return r, self.terminal

        a_before = a
        a = self.action_map[a]

        # Resolve player action
        if(a=='l'):
            self.pos = max(0, self.pos-1)
        elif(a=='r'):
            self.pos = min(9,self.pos+1)

        # Update ball position
        self.last_x = self.ball_x
        self.last_y = self.ball_y
        if(self.ball_dir == 0):
            new_x = self.ball_x-1
            new_y = self.ball_y-1
        elif(self.ball_dir == 1):
            new_x = self.ball_x+1
            new_y = self.ball_y-1
        elif(self.ball_dir == 2):
            new_x = self.ball_x+1
            new_y = self.ball_y+1
        elif(self.ball_dir == 3):
            new_x = self.ball_x-1
            new_y = self.ball_y+1

        strike_toggle = False
        if(new_x<0 or new_x>9):
            if(new_x<0):
                new_x = 0
            if(new_x>9):
                new_x=9
            self.ball_dir=[1,0,3,2][self.ball_dir]
        if(new_y<0):
            new_y = 0
            self.ball_dir=[3,2,1,0][self.ball_dir]
        elif(self.brick_map[new_y,new_x]==1):
            strike_toggle = True
            if(not self.strike):
                r+=1
                self.strike = True
                self.brick_map[new_y,new_x]=0
                new_y = self.last_y
                self.ball_dir=[3,2,1,0][self.ball_dir]
        elif(new_y == 9):
            if(np.count_nonzero(self.brick_map)==0):
                self.brick_map[1:4,:] = 1
            if(self.ball_x == self.pos):
                self.ball_dir=[3,2,1,0][self.ball_dir]
                new_y = self.last_y
            elif(new_x == self.pos):
                self.ball_dir=[2,3,0,1][self.ball_dir]
                new_y = self.last_y
            else:
                self.terminal = True

        if(not strike_toggle):
            self.strike = False

        #Increment the clock
        self.clock += 1
        if self.clock == max_clock: self.terminal = True

        self.ball_x = new_x
        self.ball_y = new_y
        return r, self.terminal

    # gets a random int in [min, max)
    def _randint(self, min, max):
        self.seed += seeded_randint(self.seed, 0, 1000)
        return seeded_randint(self.seed, min, max)

    # Query the current level of the difficulty ramp, difficulty does not ramp in this game, so return None
    def difficulty_ramp(self):
        return None  

    # Process the game-state into the 10x10xn state provided to the agent and return
    def state(self):
        state = np.zeros((10,10,len(self.channels)),dtype=bool)
        state[self.ball_y,self.ball_x,self.channels['ball']] = 1
        state[9,self.pos, self.channels['paddle']] = 1
        state[self.last_y,self.last_x,self.channels['trail']] = 1
        state[:,:,self.channels['brick']] = self.brick_map
        return state

    # Reset to start state for new episode
    def reset(self, seed=None):
        if seed is None: seed = np.random.randint(0, 10000)
        self.seed = seed
        self.clock = 0
        self.ball_y = 3
        ball_start = self._randint(0, 2)
        self.ball_x, self.ball_dir = [(0,2),(9,3)][ball_start]
        self.pos = self._randint(0, 10)
        self.brick_map = np.zeros((10,10))
        self.brick_map[1:4,:] = 1
        self.strike = False
        self.last_x = self.ball_x
        self.last_y = self.ball_y
        self.terminal = False

    # take a snapshot that can be restored from later
    def snapshot(self):
        return np.array(
                [self.seed, self.clock, self.ball_y, self.ball_x, self.ball_dir,
                self.pos] + self.brick_map.reshape(-1).tolist() + [self.strike,
                self.last_x, self.last_y, self.terminal], dtype=np.int)

    # restore from a snapshot
    def restore(self, snapshot):
        self.seed, self.clock, self.ball_y, self.ball_x, self.ball_dir, self.pos = snapshot[:6]
        self.brick_map = snapshot[6:106].reshape((10,10))
        self.strike, self.last_x, self.last_y, self.terminal = snapshot[106:]
        self.strike = bool(self.strike)
        self.terminal = bool(self.terminal)

    # Dimensionality of the game-state (10x10xn)
    def state_shape(self):
        return [10,10,len(self.channels)]

    # Number of ints in the snapshot
    def snapshot_size(self):
        return 110

    # Range of per-transition rewards
    def reward_range(self):
        return (0, 1)

    # Subset of actions that actually have a unique impact in this environment
    def minimal_action_set(self):
        minimal_actions = ['n','l','r']
        return [self.action_map.index(x) for x in minimal_actions]

class VectorizedEnv(Env):
    def __init__(self, num_envs=None, seeds=None, ramping = None):
        self.channels ={
            'paddle':0,
            'ball':1,
            'trail':2,
            'brick':3,
        }
        self.action_map = ['n','l','u','r','d','f']
        self.inverse_action_map = {self.action_map[i]: i for i in range(len(self.action_map))}
        if num_envs is not None or seeds is not None: self.reset(num_envs, seeds)

    # Reset to start state for new episode
    def reset(self, num_envs=None, seeds=None):
        if num_envs is None:
            assert seeds is not None
            self.num_envs = len(seeds)
        else:
            self.num_envs = num_envs
        if seeds is None:
            seeds = np.random.randint(0, 10000, [self.num_envs])
        self.seeds = seeds
        self.clock = np.zeros(self.num_envs)
        self.ball_y = np.full(self.num_envs, 3)
        ball_start = self._randint(0, 2)
        ball_x_and_dir = np.array([(0,2),(9,3)])[ball_start]
        self.ball_x, self.ball_dir = ball_x_and_dir[:,0], ball_x_and_dir[:,1]
        self.pos = self._randint(0, 10)
        self.brick_map = np.zeros((self.num_envs,10,10))
        self.brick_map[:,1:4,:] = 1
        self.strike = np.full(self.num_envs, False)
        self.last_x = self.ball_x
        self.last_y = self.ball_y
        self.terminal = np.full(self.num_envs, False)

    def act(self, a):
        r = np.zeros(self.num_envs)
        already_terminal = self.terminal[:]

        self.pos[a == self.inverse_action_map['l']] -= 1
        self.pos[a == self.inverse_action_map['r']] += 1
        self.pos = np.clip(self.pos, 0, 9)

        # Update ball position
        self.last_x = self.ball_x
        self.last_y = self.ball_y
        new_x = self.last_x[:]
        new_y = self.last_y[:]
        new_x[self.ball_dir == 0] -= 1
        new_x[self.ball_dir == 1] += 1
        new_x[self.ball_dir == 2] += 1
        new_x[self.ball_dir == 3] -= 1
        new_y[self.ball_dir == 0] -= 1
        new_y[self.ball_dir == 1] -= 1
        new_y[self.ball_dir == 2] += 1
        new_y[self.ball_dir == 3] += 1

        strike_toggle = np.full(self.num_envs, False)

        wall_bounce_locs = (new_x < 0) | (new_x > 9)
        top_collision_locs = new_y < 0
        bottom_collision_locs = new_y == 9

        new_x = np.clip(new_x, 0, 9)
        new_y = np.clip(new_y, 0, 9)
        new_ball_pos = (np.arange(self.num_envs), new_y, new_x)
        brick_collision_locs = self.brick_map[new_ball_pos] == 1
        brickless_locs = np.count_nonzero(self.brick_map, axis=(1,2)) == 0
        paddle_collision_locs = bottom_collision_locs & (self.ball_x == self.pos)
        edge_paddle_collision_locs = bottom_collision_locs & (new_x == self.pos) & (~paddle_collision_locs)

        self.ball_dir[wall_bounce_locs] = np.array([1,0,3,2])[self.ball_dir[wall_bounce_locs]]
        self.ball_dir[top_collision_locs] = np.array([3,2,1,0])[self.ball_dir[top_collision_locs]]

        strike_toggle[brick_collision_locs] = True
        new_strike_locs = brick_collision_locs & (~self.strike)
        r[new_strike_locs] += 1
        self.strike[new_strike_locs] = True
        self.brick_map[tuple([item[new_strike_locs] for item in new_ball_pos])] = 0
        new_y[new_strike_locs] = self.last_y[new_strike_locs]
        self.ball_dir[new_strike_locs] = np.array([3,2,1,0])[self.ball_dir[new_strike_locs]]

        self.brick_map[(bottom_collision_locs & brickless_locs),1:4,:] = 1
        self.ball_dir[paddle_collision_locs] = np.array([3,2,1,0])[self.ball_dir[paddle_collision_locs]]
        new_y[paddle_collision_locs] = self.last_y[paddle_collision_locs]
        self.ball_dir[edge_paddle_collision_locs] = np.array([2,3,0,1])[self.ball_dir[edge_paddle_collision_locs]]
        new_y[edge_paddle_collision_locs] = self.last_y[edge_paddle_collision_locs]
        self.terminal[(bottom_collision_locs & ~(paddle_collision_locs | edge_paddle_collision_locs))] = True

        r[already_terminal] = 0.

        self.strike[~strike_toggle] = False

        #Increment the clock
        self.clock += 1
        self.terminal[self.clock == max_clock] = True

        self.ball_x = new_x
        self.ball_y = new_y
        return r, self.terminal

    def state(self):
        state = np.zeros((10,10,len(self.channels)),dtype=bool)
        state[self.ball_y,self.ball_x,self.channels['ball']] = 1
        state[9,self.pos, self.channels['paddle']] = 1
        state[self.last_y,self.last_x,self.channels['trail']] = 1
        state[:,:,self.channels['brick']] = self.brick_map
        return state

    def restore(self, snapshot_batch):
        self.seed, self.clock, self.ball_y, self.ball_x, self.ball_dir, self.pos = snapshot[:6]
        self.brick_map = snapshot[6:106].reshape((10,10))
        self.strike, self.last_x, self.last_y, self.terminal = snapshot[106:]
        self.strike = bool(self.strike)
        self.terminal = bool(self.terminal)

    def _randint(self, min, max):
        for i, seed in enumerate(self.seeds):
            self.seeds[i] += seeded_randint(seed, 0, 1000)
        return np.array([seeded_randint(seed, min, max) for seed in self.seeds])
