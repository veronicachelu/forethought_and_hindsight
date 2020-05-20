from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import dm_env
from dm_env import specs
import numpy as np
import os
from utils.mdp_solvers.solve_mdp import MdpSolver
from basis.tile import *
from utils.visualizer import *
class Actions():
    up = 0
    right = 1
    down = 2
    left = 3


class ObstacleWorld(dm_env.Environment):
    def __init__(self, path=None, stochastic=False, random_restarts=False,
                 rng=None, env_size=None, obs_type=None):
        self._str_MDP = ''
        self._height = -1
        self._width = -1
        self._path = path

        self._cPos = np.zeros((2,))

        self._read_file(path)
        self._parse_string()
        self._stochastic = stochastic
        self._nA = 4
        self._rng = rng
        self._reset_next_step = True

        self._P = None
        self._P_absorbing = None
        self._R = None
        self._d = None

        self._obs_type = obs_type

        self._starting_positions = []
        for si in range(self._height):
            for sj in range(self._width):
                sPos = np.array([si, sj])
                if not self._is_obstacle(sPos) and not self._is_terminal(sPos):
                    self._starting_positions.append(sPos)

        self._nS = self._height * self._width

        self._fill_P_R()

    def _read_file(self, path):
        file = open(path, 'r')
        for line in file:
            self._str_MDP += line

    def _parse_string(self):
        data = self._str_MDP.split('\n')
        self._height = int(data[1].split(' ')[0])
        self._width = int(data[1].split(' ')[1])

        self._step_size = int(data[3].split(' ')[0])
        self._mean_step_size = float(data[3].split(' ')[1])
        self._var_step_size = float(data[3].split(' ')[2])

        self._rewards = []
        for reward in np.arange(5, 8):
            gX = int(data[reward].split(' ')[0])
            gY = int(data[reward].split(' ')[1])
            reward_val = float(data[reward].split(' ')[2])
            reward_noise = float(data[reward].split(' ')[3])
            self._rewards.append((gX, gY, reward_val, reward_noise))
            break

        self._obstacles = []
        for obstacle in np.arange(9, len(data)):
            xsO = int(data[obstacle].split(' ')[0])
            xfO = int(data[obstacle].split(' ')[1])
            ysO = int(data[obstacle].split(' ')[2])
            yfO = int(data[obstacle].split(' ')[3])
            self._obstacles.append((xsO, xfO, ysO, yfO))

    def _is_obstacle(self, pos):
        isit = False
        for obstacle in self._obstacles:
            if pos[0] >= obstacle[0] and pos[0] < obstacle[1] and \
                pos[1] >= obstacle[2] and pos[1] < obstacle[3]:
                isit = True
        return isit

    def reset(self):
        """Returns the first `TimeStep` of a new episode."""
        self._reset_next_step = False
        s_i = self._rng.choice(np.arange(len(self._starting_positions)),
                         p=[1/len(self._starting_positions) for _ in self._starting_positions])

        self._cPos = self._starting_positions[s_i]
        return dm_env.restart(self._observation())

    def _is_terminal(self, pos):
        for reward in self._rewards:
            gX, gY, reward_val, reward_noise = reward
            gPos = np.array([gX, gY])
            if np.linalg.norm(pos - gPos) < self._step_size:
                return True

        return False

    def _get_reward(self, pos):
        total_r = 0.0
        for reward in self._rewards:
            gX, gY, reward_val, reward_noise = reward
            gPos = np.array([gX, gY])
            if np.linalg.norm(pos - gPos) < self._step_size:
                total_r += self._rng.normal(loc=reward_val, scale=reward_noise)

        return total_r

    # def _gaussian1d(self, p, mu, sig):
    #     return np.exp(-((p - mu) ** 2) / (2. * sig ** 2)) / (sig * np.sqrt(2. * np.pi))

    def _take_action(self, action):
        potential_pos = np.copy(self._cPos)

        DIR_TO_VEC = [
            # up
            np.array((-1, 0)),
            # right
            np.array((0, 1)),
            # down
            np.array((1, 0)),
            # left
            np.array((0, -1)),
        ]

        if self._stochastic:
            random_move = self._rng.choice(range(2), p=[0.9, 0.1])
            if random_move:
                action = self._rng.choice(range(self._nA),
                                          p=[1/self._nA for _ in range(self._nA)])

        step_size = DIR_TO_VEC[action] * self._step_size
        if self._stochastic:
            step_size += self._rng.normal(loc=self._mean_step_size,
                                         scale=self._var_step_size, size=(2,))

        potential_pos += step_size
        potential_pos = potential_pos.clip([0, 0], [self._height - self._step_size,
                                                    self._width - self._step_size])

        if not self._is_obstacle(potential_pos):
            self._cPos = np.copy(potential_pos)

        reward = self._get_reward(self._cPos)

        return reward

    def step(self, action):
        """Updates the environment according to the action."""
        if self._reset_next_step:
            return self.reset()

        reward = self._take_action(action)
        if self._is_terminal(self._cPos):
            self._reset_next_step = True
            return dm_env.termination(reward=reward, observation=self._observation())
        return dm_env.transition(reward=reward, observation=self._observation())
        # return dm_env.transition(reward=reward, observation=self._observation())

    def _pos_to_idx(self, pos):
        return np.ravel_multi_index(pos, (self._height, self._width))

    def _observation(self):
        if self._obs_type == "position":
            return self._cPos

    def observation_spec(self):
        if self._obs_type == "position":
            return specs.BoundedArray(shape=(2,), dtype=np.int32,
                                      name="state", minimum=0, maximum=1)

    def action_spec(self):
        return specs.DiscreteArray(
            dtype=int, num_values=self._nA, name="action")

    def _fill_d(self):
        self._d = np.zeros((self._nS), dtype=np.float)

        for sPos in self._starting_positions:
            self._d[self._pos_to_idx(sPos)] = 1 / len(
                self._starting_positions)

    def _fill_P_R(self):
        self._P = np.zeros((self._nA, self._nS, self._nS), dtype=np.float)
        self._P_absorbing = np.zeros((self._nA, self._nS, self._nS), dtype=np.float)
        self._R = np.zeros((self._nA, self._nS, self._nS), dtype=np.float)
        self._d = np.zeros((self._nS), dtype=np.float)

        DIR_TO_VEC = [
            # up
            np.array((-1, 0)),
            # right
            np.array((0, 1)),
            # down
            np.array((1, 0)),
            # left
            np.array((0, -1)),
        ]

        for sPos in self._starting_positions:
            self._d[self._pos_to_idx(sPos)] = 1 / len(
                                self._starting_positions)

        for i in range(self._height):
            for j in range(self._width):
                for k in range(self._nA):
                    pos = np.array([i, j])
                    if self._is_terminal(pos):
                        for sPos in self._starting_positions:
                            self._P[k][self._pos_to_idx(pos)][self._pos_to_idx(sPos)] = 1 / len(
                                self._starting_positions)
                        self._P_absorbing[k][self._pos_to_idx(pos)][self._pos_to_idx(pos)] = 1
                        self._R[k][self._pos_to_idx(pos)][self._pos_to_idx(pos)] = 0
                    elif not self._is_obstacle(pos):
                        step_size = DIR_TO_VEC[k] * self._step_size
                        fwd_pos = pos + step_size
                        fwd_i, fwd_j = fwd_pos
                        if fwd_i >= 0 and fwd_i < self._height and \
                                fwd_j >= 0 and fwd_j < self._width and not self._is_obstacle(fwd_pos):
                            self._P[k][self._pos_to_idx(pos)][self._pos_to_idx(fwd_pos)] = 1
                            self._P_absorbing[k][self._pos_to_idx(pos)][self._pos_to_idx(fwd_pos)] = 1
                            self._R[k][self._pos_to_idx(pos)][self._pos_to_idx(fwd_pos)] = \
                                        self._get_reward(fwd_pos)
                        else:
                            self._P[k][self._pos_to_idx(pos)][self._pos_to_idx(pos)] = 1
                            self._P_absorbing[k][self._pos_to_idx(pos)][self._pos_to_idx(pos)] = 1
                            self._R[k][self._pos_to_idx(pos)][self._pos_to_idx(pos)] = 0
                    # else:
                        # self._P[k][self._pos_to_idx(pos)][self._pos_to_idx(pos)] = 1
                        # self._P_absorbing[k][self._pos_to_idx(pos)][self._pos_to_idx(pos)] = 1
                        # self._R[k][self._pos_to_idx(pos)][self._pos_to_idx(pos)] = 0

    def _get_dynamics(self, feature_coder=None):
        if self._P is None or self._R is None or self._P_absorbing is None:
            self._fill_P_R()

        return self._P, self._P_absorbing, self._R, self._d

    def reshape_v(self, v):
        return np.reshape(v, (self._height, self._width))

    def reshape_pi(self, pi):
        return np.reshape(pi, (self._feature_extractor_h, self._feature_extractor_w, self._nA))

    def get_all_states(self):
        states = []
        for i in range(self._height):
            for j in range(self._width):
                pos = np.array([i, j])
                states.append(pos)
        return states

if __name__ == "__main__":
    nrng = np.random.RandomState(0)
    nS = None
    nA = 4
    discount = 0.99
    mdp_filename = "../../continuous_mdps/obstacle.mdp"
    env = ObstacleWorld(path=mdp_filename, stochastic=False,
                        rng=nrng, env_size=None)
    nS = env._nS
    nA = 4

    # plot_grid(env, logs=str(os.environ['LOGS']), env_type="continuous")
    mdp_solver = MdpSolver(env, nS, nA, discount)
    v = mdp_solver.get_optimal_v()
    # v = env.reshape_v(v)
    # pi = mdp_solver.get_optimal_policy()
    # policy = lambda x, nrng: np.argmax(pi[x])

    # mdp_solver = MdpSolver(env, nS, nA, discount)
    # # policy = mdp_solver.get_optimal_policy()
    # v = mdp_solver.get_optimal_v()
    v = env.reshape_v(v)
    plot_v(env, v, str(os.environ['LOGS']), env_type="continuous")
    # plot_policy(env, env.reshape_pi(env._pi), str((os.environ['LOGS'])),
    #             env_type="continuous")
    pi = np.full((env._nS, env._nA), 1 / env._nA)
    # eta_pi = mdp_solver.get_eta_pi(pi)

    plot_error(env, v, str(os.environ['LOGS']), env_type="continuous",
               eta_pi=env._d,
               filename="v_eta.png")


    # plot_eta_pi(env, env.reshape_v(eta_pi))