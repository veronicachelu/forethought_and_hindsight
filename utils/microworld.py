from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import dm_env
from dm_env import specs
import numpy as np
import scipy

class Actions():
    up = 0
    right = 1
    down = 2
    left = 3


class MicroWorld(dm_env.Environment):
    def __init__(self, path=None, stochastic=False, random_restarts=False,
                 seed=0, rng=None, obs_type="tabular", env_size=1, max_reward=1):
        self._str_MDP = ''
        self._height = -1
        self._width = -1
        self._nS = -1
        self._mdp = None
        self._adj_matrix = None
        self._nX = env_size
        self._max_reward = max_reward

        self._cX, self._cY = 0, 0
        self._sX, self._sY = 0, 0
        self._gX, self._gY = 0, 0

        self._read_file(path)
        self._parse_string()
        self._stochastic = stochastic
        self._random_restarts = random_restarts
        self._p = 0.2
        self._cX = self._sX
        self._cY = self._sY
        self._nS = self._height * self._width
        self._nA = 4
        self._rng = rng
        self._reset_next_step = True
        self._obs_type = obs_type


    def _read_file(self, path):
        file = open(path, 'r')
        for line in file:
            self._str_MDP += line

    def _parse_string(self):
        data = self._str_MDP.split('\n')
        self._height = int(data[0].split(',')[0]) #* self._nX
        self._width = int(data[0].split(',')[1]) #* self._nX
        self._mdp = np.zeros((self._height, self._width))

        for i in np.arange(0, len(data) - 1):
            for j in np.arange(len(data[i + 1])):
                if data[i+1][j] == 'X':
                    # for kx in range(self._nX):
                    #     for ky in range(self._nX):
                    #         self._mdp[i * self._nX + kx][j * self._nX + ky] = -1
                    self._mdp[i][j] = -1
                elif data[i+1][j] == '.':
                    # for kx in range(self._nX):
                    #     for ky in range(self._nX):
                    #         self._mdp[i * self._nX + kx][j * self._nX + ky] = 0
                    self._mdp[i][j] = 0
                elif data[i+1][j] == 'S':
                    self._sX = i# * self._nX
                    self._sY = j# * self._nX
                elif data[i+1][j] == 'G':
                    self._gX = i# * self._nX
                    self._gY = j# * self._nX
                    # self._mdp[ self._gX][self._gY] = 0


    def _get_state_index(self, x, y):
        idx = y + x * self._width
        return idx

    def _get_state_coords(self, idx):
        y = idx % self._width
        x = (idx - y) / self._height

        return int(x), int(y)

    def _get_crt_state(self):
        s = self._get_state_index(self._cX, self._cY)
        return s

    def _get_goal_state(self):
        g = self._get_state_index(self._gX, self._gY)
        return g

    def _get_next_reward(self, nX, nY):
        if nX == self._gX and nY == self._gY:
            return self._max_reward
        else:
            return 0

    def _is_terminal(self):
        return self._cX == self._gX and self._cY == self._gY

    def _define_goal_state(self, g):
        x, y = self._get_state_coords(g)

        if g >= self._nS:
            return False
        elif self._mdp[x][y] == -1:
            return False
        else:
            self._gX = x
            self._gY = y
            self.reset()
            return True

    def get_next_state_and_reward(self, s, action):
        if s == self._nS:
            return s, 0

        s = self._get_crt_state()
        tempX = self._cX
        tempY = self._cY
        self._cX, self._cY = self._get_state_coords(s)

        if self._is_terminal():
            next_s = self._nS
            reward = 0
        else:
            nX, nY = self._get_next_state(action)
            if nX != -1 and nY != -1:  # If it is not the absorbing state:
                reward = self._get_next_reward(nX, nY)
                next_s = self._get_state_index(nX, nY)
            else:
                reward = 0
                next_s = self._nS

        self._cX = tempX
        self._cY = tempY

        return next_s, reward

    def _get_next_state(self, action):
        nX = self._cX
        nY = self._cY

        self._possible_next_states = []
        if self._cX > 0 and self._mdp[self._cX - 1][self._cY] != -1:
            self._possible_next_states.append((self._cX - 1, self._cY))
        else:
            self._possible_next_states.append((self._cX, self._cY))
        if self._cY < self._width - 1 and self._mdp[self._cX][self._cY + 1] != -1:
            self._possible_next_states.append((self._cX, self._cY + 1))
        else:
            self._possible_next_states.append((self._cX, self._cY))
        if self._cX < self._height - 1 and self._mdp[self._cX + 1][self._cY] != -1:
            self._possible_next_states.append((self._cX + 1, self._cY))
        else:
            self._possible_next_states.append((self._cX, self._cY))
        if self._cY > 0 and self._mdp[self._cX][self._cY - 1] != -1:
            self._possible_next_states.append((self._cX, self._cY - 1))
        else:
            self._possible_next_states.append((self._cX, self._cY))

        if self._mdp[self._cX][self._cY] != -1:
            if action == Actions.up and self._cX > 0:
                nX = self._cX - 1
                nY = self._cY
            elif action == Actions.right and self._cY < self._width - 1:
                nX = self._cX
                nY = self._cY + 1
            elif action == Actions.down and self._cX < self._height - 1:
                nX = self._cX + 1
                nY = self._cY
            elif action == Actions.left and self._cY > 0:
                nX = self._cX
                nY = self._cY - 1

        if self._stochastic:
            p = [self._p, 1 - self._p]
            random_move = self._rng.choice(a=np.arange(4),
                             p=[1/4] * 4)
            next_states = [self._possible_next_states[random_move], (nX, nY)]
            next_state = self._rng.choice([0, 1], p=p)
            next_state = next_states[next_state]
        else:
            next_state = (nX, nY)
        if self._mdp[next_state[0]][next_state[1]] != -1:
            return next_state
        else:
            return self._cX, self._cY

    def reset(self):
        """Returns the first `TimeStep` of a new episode."""
        self._reset_next_step = False
        if self._random_restarts:
            valid = False
            while not valid:
                self._sX = self._rng.randint(self._height)
                self._sY = self._rng.randint(self._width)
                if self._mdp[self._sX][self._sY] != -1 and \
                        (not (self._sX == self._gX and self.sY == self._gY)):
                    valid = True
        self._cX = self._sX
        self._cY = self._sY
        return dm_env.restart(self._observation())

    def step(self, action):
        """Updates the environment according to the action."""
        if self._reset_next_step:
            return self.reset()

        nX, nY = self._get_next_state(action)
        reward = self._get_next_reward(nX, nY)
        self._cX = nX
        self._cY = nY

        if self._is_terminal():
            self._reset_next_step = True
            return dm_env.termination(reward=reward, observation=self._observation())
        return dm_env.transition(reward=reward, observation=self._observation())


    def observation_spec(self):
        if self._obs_type == "tabular":
            return specs.BoundedArray(shape=(self._nS,), dtype=np.int32,
                                  name="state", minimum=0, maximum=self._nS)
        elif self._obs_type == "onehot":
            return specs.BoundedArray(shape=(self._nS,), dtype=np.int32,
                                  name="state", minimum=0, maximum=1)
        elif self._obs_type == "pixels":
            return specs.BoundedArray(shape=self._mdp.shape, dtype=np.float32,
                                  name="state", minimum=0, maximum=1)

    def action_spec(self):
        return specs.DiscreteArray(
            dtype=int, num_values=self._nA, name="action")

    def _observation(self):
        if self._obs_type == "tabular":
            return self._get_state_index(self._cX, self._cY)
        elif self._obs_type == "onehot":
            return np.eye(self._nS)[self._get_state_index(self._cX, self._cY)]
        elif self._obs_type == "pixels":
            board = np.zeros((self._height, self._width), dtype=np.float32)
            board[self._mdp == -1] = 0.5
            board[self._cX][self._cY] = 1
            return board
