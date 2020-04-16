from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import dm_env
from dm_env import specs
import numpy as np
from basis.tile import *
class Actions():
    up = 0
    right = 1
    down = 2
    left = 3


class PuddleWorld(dm_env.Environment):
    def __init__(self, path=None, stochastic=False, random_restarts=False,
                 rng=None, obs_type="tile", env_size=16):
        self._str_MDP = ''
        self._height = -1
        self._width = -1
        self._path = path

        self._cPos = np.zeros((2,))
        self._sPos = np.zeros((2,))
        self._gPos = np.zeros((2,))

        self._read_file(path)
        self._parse_string()
        self._stochastic = stochastic
        self._random_restarts = random_restarts
        self._p = 0.1
        self._cPos[0] = self._sPos[0]
        self._cPos[1] = self._sPos[1]
        self._nA = 4
        self._rng = rng
        self._reset_next_step = True
        self._obs_type = obs_type
        self._low = [0.0, 0.0]
        self._high = [self._height, self._width]
        self._bins = env_size ** 2
        self._bins_dim = env_size

        self._P = None
        self._R = None

        self._tilings = [create_tiling_grid(self._low, self._high,
                                           bins=(self._bins_dim, self._bins_dim),
                                           offsets=(0.0, 0.0))]
        self._h = int(self._height // self._step_size)
        self._w = int(self._width // self._step_size)

        self._mdp = np.zeros((self._h, self._w))

        self._hh = self._bins_dim
        self._ww = self._bins_dim
        self._nS = self._hh * self._ww
        self._mdp_tilings = np.zeros((self._hh, self._ww))

        for obstacle in self._obstacles:
            xsO, xfO, ysO, yfO = obstacle
            for i in range(int(xsO//self._step_size),
                           int(xfO // self._step_size)):
                for j in range(int(ysO//self._step_size),
                               int(yfO//self._step_size)):
                    self._mdp[i][j] = -1
                    encoded_obs = tile_encode((i * self._step_size, j * self._step_size), self._tilings)[0]
                    self._mdp_tilings[encoded_obs[0]][encoded_obs[1]] = -1


        self._sX = int(self._sPos[0]//self._step_size)
        self._sY = int(self._sPos[1]//self._step_size)
        self._sX_tilings, self._sY_tilings = tile_encode(self._sPos, self._tilings)[0]
        self._sX_tilings = self._sX_tilings
        self._g = [(int(self._gPos[0]//self._step_size),
                    int(self._gPos[1]//self._step_size)
                    )]
        self._g_tilings = [tile_encode(self._gPos, self._tilings)[0]]
        self._g_tilings = [(self._g_tilings[0][0], self._g_tilings[0][1])]

    def _read_file(self, path):
        file = open(path, 'r')
        for line in file:
            self._str_MDP += line

    def _parse_string(self):
        data = self._str_MDP.split('\n')
        self._height = float(data[1].split(' ')[0])
        self._width = float(data[1].split(' ')[1])

        self._step_size = float(data[3].split(' ')[0])
        self._mean_step_size = float(data[3].split(' ')[1])
        self._var_step_size = float(data[3].split(' ')[2])

        self._sPos[0] = float(data[5].split(' ')[0])
        self._sPos[1] = float(data[5].split(' ')[1])

        self._gPos[0] = float(data[7].split(' ')[0])
        self._gPos[1] = float(data[7].split(' ')[1])
        self._fudge = float(data[7].split(' ')[2])
        self._reward = float(data[7].split(' ')[3])
        self._reward_noise = float(data[7].split(' ')[4])

        self._obstacles = []
        for obstacle in np.arange(9, len(data)):
            xsO = float(data[obstacle].split(' ')[0])
            xfO = float(data[obstacle].split(' ')[1])
            ysO = float(data[obstacle].split(' ')[2])
            yfO = float(data[obstacle].split(' ')[3])
            self._obstacles.append((xsO, xfO, ysO, yfO))

    def _is_obstacle(self, pos):
        isit = False
        for obstacle in self._obstacles:
            if pos[0] >= obstacle[0] and pos[0] < obstacle[1] and \
                pos[1] >= obstacle[1] and pos[1] < obstacle[2]:
                isit = True
        return isit

    def reset(self):
        """Returns the first `TimeStep` of a new episode."""
        self._reset_next_step = False
        if self._random_restarts:
            valid = False
            while not valid:
                self._sPos[0] = self._rng.randint(self._height)
                self._sPos[1] = self._rng.randint(self._width)
                if (not self._is_obstacle(self._sPos)) and \
                        (not np.linalg.norm(self._sPos - self._gPos) < self._fudge):
                    valid = True
        self._cPos[0] = self._sPos[0]
        self._cPos[1] = self._sPos[1]
        return dm_env.restart(self._observation())

    def _take_action(self, action):
        potential_pos = np.copy(self._cPos)
        if action == Actions.up:
            potential_pos[0] -= self._step_size
        elif action == Actions.right:
            potential_pos[1] += self._step_size
        elif action == Actions.down:
            potential_pos[0] += self._step_size
        elif action == Actions.left:
            potential_pos[1] -= self._step_size

        if self._stochastic:
            potential_pos += self._rng.normal(loc=self._mean_step_size,
                                         scale=self._var_step_size, size=(2,))
        potential_pos = potential_pos.clip([0, 0], [self._height, self._width])

        if not self._is_obstacle(potential_pos):
            self._cPos = np.copy(potential_pos)

        return self._reward if self._is_terminal() else 0.0

    def step(self, action):
        """Updates the environment according to the action."""
        if self._reset_next_step:
            return self.reset()

        reward = self._take_action(action)

        if self._stochastic:
            reward += self._rng.normal(scale=self._reward_noise)

        if self._is_terminal():
            self._reset_next_step = True
            return dm_env.termination(reward=reward, observation=self._observation())
        return dm_env.transition(reward=reward, observation=self._observation())

    def _is_terminal(self):
        return np.linalg.norm(self._cPos - self._gPos) < self._fudge

    def _observation(self):
        if self._obs_type == "position":
            return self._cPos
        elif self._obs_type == "tile":
            encoded_obs = tile_encode(self._cPos, self._tilings)[0]
            index = np.ravel_multi_index(encoded_obs, (self._bins_dim, self._bins_dim))
            onehotstate = np.eye(self._bins)[index]
            return onehotstate

    def observation_spec(self):
        if self._obs_type == "position":
            return specs.BoundedArray(shape=(2,), dtype=np.int32,
                                      name="state", minimum=0, maximum=1)
        elif self._obs_type == "tile":
            return specs.BoundedArray(shape=(self._bins,), dtype=np.int32,
                                      name="state", minimum=0, maximum=1)

    def action_spec(self):
        return specs.DiscreteArray(
            dtype=int, num_values=self._nA, name="action")

    def _fill_P_R(self):
        self._P = np.zeros((self._nA, self._nS, self._nS), dtype=np.float)
        self._P_absorbing = np.zeros((self._nA, self._nS, self._nS), dtype=np.float)
        self._R = np.zeros((self._nA, self._nS, self._nS), dtype=np.float)
        self._index_matrix = np.zeros((self._h, self._w), dtype=np.float)

        for i in range(self._hh):
            for j in range(self._ww):
                self._index_matrix[i][j] = np.ravel_multi_index((i, j), (self._hh, self._ww))

        DIR_TO_VEC = [
            # up
            np.array((-1, 0)),
            # right
            np.array((0, 1)),
            # down
            np.array((1, 0)),
            #left
            np.array((0, -1)),
        ]
        for i in range(self._hh):
            for j in range(self._ww):
                for k in range(self._nA):
                    fwd_pos = np.array([i, j]) + DIR_TO_VEC[k]
                    fwd_i, fwd_j = fwd_pos
                    if self._mdp_tilings[i][j] != -1:
                        if not ((i, j) in self._g_tilings):
                            if fwd_i >= 0 and fwd_i < self._h and\
                                fwd_j >= 0 and fwd_j < self._w and self._mdp[fwd_i][fwd_j] != -1:
                                self._P[k][self._index_matrix[i][j]][self._index_matrix[fwd_i][fwd_j]] = 1
                                self._P_absorbing[k][self._index_matrix[i][j]][self._index_matrix[fwd_i][fwd_j]] = 1
                                self._R[k][self._index_matrix[i][j]][self._index_matrix[fwd_i][fwd_j]] = \
                                    self._reward if (fwd_i, fwd_j) in self._g_tilings else 0
                                self._P[k][self._index_matrix[i][j]][self._index_matrix[i][j]] = 0
                                self._P_absorbing[k][self._index_matrix[i][j]][self._index_matrix[i][j]] = 0
                                self._R[k][self._index_matrix[i][j]][self._index_matrix[i][j]] = \
                                    self._reward if (i, j) in self._g_tilings else 0
                            else:
                                self._P[k][self._index_matrix[i][j]][self._index_matrix[i][j]] = 1
                                self._P_absorbing[k][self._index_matrix[i][j]][self._index_matrix[i][j]] = 1
                                self._R[k][self._index_matrix[i][j]][self._index_matrix[i][j]] = \
                                    self._reward if (i, j) in self._g_tilings else 0
                        else:
                            self._P[k][self._index_matrix[i][j]][self._index_matrix[self._sX_tilings][self._sY_tilings]] = 1
                            self._P_absorbing[k][self._index_matrix[i][j]][self._index_matrix[i][j]] = 1
                            self._R[k][self._index_matrix[i][j]][self._index_matrix[i][j]] = 0
                            # \
                            #     self._reward if (i, j) in self._g_tilings else 0

    def _get_dynamics(self):
        if self._P == None or self._R == None or self._P_absorbing == None:
            self._fill_P_R()

        return self._P, self._P_absorbing, self._R

    def reshape_v(self, v):
        return np.reshape(v, (self._hh, self._ww))

    def reshape_pi(self, pi):
        return np.reshape(pi, (self._hh, self._ww, self._nA))

    def get_all_states(self):
        states = []
        for i in range(self._mdp_tilings.shape[0]):
            for j in range(self._mdp_tilings.shape[1]):
                index = np.ravel_multi_index((i, j), (self._bins_dim, self._bins_dim))
                onehotstate = np.eye(self._bins)[index]
                states.append(onehotstate)
        return states
