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


class ContinuousWorld(dm_env.Environment):
    def __init__(self, path=None, stochastic=False, random_restarts=False,
                 seed=0, rng=None, obs_type="tile", env_size=16):
        self._str_MDP = ''
        self._height = -1
        self._width = -1

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

        self._tilings = [create_tiling_grid(self._low, self._high,
                                           bins=(self._bins_dim, self._bins_dim),
                                           offsets=(0.0, 0.0))]

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
            xsO = data[obstacle].split(' ')[0]
            xfO = data[obstacle].split(' ')[1]
            ysO = data[obstacle].split(' ')[2]
            yfO = data[obstacle].split(' ')[3]
            self._obstacles.append((xsO, xfO, ysO, yfO))

    def _is_obstacle(self, pos):
        ok = True
        for obstacle in self._obstacles:
            if pos[0] >= obstacle[0] and pos[0] < obstacle[0] and \
                pos[1] >= obstacle[1] and pos[1] > obstacle[1]:
                    ok = False
        return ok

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
        if action == Actions.up:
            self._cPos[0] += self._step_size
        elif action == Actions.right:
            self._cPos[1] += self._step_size
        elif action == Actions.down:
            self._cPos[0] -= self._step_size
        elif action == Actions.left:
            self._cPos[1] -= self._step_size

        if self._stochastic:
            self._cPos += np.random.normal(loc=self._mean_step_size,
                                         scale=self._var_step_size, size=(2,))
        self._cPos = self._cPos.clip([0, 0], [self._height, self._width])

        return self._reward if self._is_terminal() else 0.0

    def step(self, action):
        """Updates the environment according to the action."""
        if self._reset_next_step:
            return self.reset()

        reward = self._take_action(action)

        if self._stochastic:
            reward += np.random.normal(scale=self._reward_noise)

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
