from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import dm_env
from dm_env import specs
import numpy as np
import scipy
from utils.mdp_solvers.solve_mdp import MdpSolver
from utils.visualizer import *

class Actions():
    up = 0
    right = 1
    down = 2
    left = 3


class MicroWorld(dm_env.Environment):
    def __init__(self, path=None, stochastic=False, random_restarts=False,
                 rng=None, obs_type="tabular", env_size=1, max_reward=1.0):
        self._str_MDP = ''
        self._height = -1
        self._width = -1
        self._nS = -1
        self._mdp = None
        self._adj_matrix = None
        self._nX = env_size
        self._max_reward = max_reward

        self._cX, self._cY = 0, 0
        # self._sX, self._sY = 0, 0
        self._g = []
        self._starting_states = []

        self._path = path
        self._read_file(path)
        self._parse_string()
        self._stochastic = stochastic
        self._random_restarts = random_restarts
        self._slip_prob = 0.5
        # self._cX = self._sX
        # self._cY = self._sY
        self._nS = self._height * self._width
        self._nA = 4
        self._rng = rng
        self._reset_next_step = True
        self._obs_type = obs_type
        self._P = None
        self._P_absorbing = None
        self._R = None
        self._d = None
        self._fill_P_R()

    def _read_file(self, path):
        file = open(path, 'r')
        for line in file:
            self._str_MDP += line

    def _parse_string(self):
        data = self._str_MDP.split('\n')
        self._height = int(data[0].split(',')[0])
        self._width = int(data[0].split(',')[1])
        self._mdp = np.zeros((self._height, self._width))

        for i in np.arange(0, len(data) - 1):
            for j in np.arange(len(data[i + 1])):
                if data[i+1][j] == 'X':
                    self._mdp[i][j] = -1
                elif data[i+1][j] == '.':
                    self._mdp[i][j] = 0
                elif data[i+1][j] == 'S':
                    self._starting_states.append((i, j))
                elif data[i+1][j] == 'G':
                    self._g.append((i, j))

    def _get_state_index(self, x, y):
        return np.ravel_multi_index((x, y), (self._height, self._width))

    def _get_state_coords(self, idx):
        return np.unravel_index(idx, (self._height, self._width))

    def _get_crt_state(self):
        s = self._get_state_index(self._cX, self._cY)
        return s

    def _get_next_reward(self, nX, nY):
        if (nX, nY) in self._g:
            return self._rng.choice([0.0, self._max_reward], p=[0.9, 0.1])#
        else:
            return 0.0

    def _is_terminal(self):
        return (self._cX, self._cY) in self._g

    def _get_next_state(self, cPos, action):
        cX, cY = cPos

        nPos = self._rng.choice(range(self._nS),
                         p=self._P[action][self._index_matrix[cX][cY]])
        nX, nY = self._get_state_coords(nPos)

        return (nX, nY)
        # nX = self._cX
        # nY = self._cY
        #
        # DIR_TO_VEC = [
        #     # up
        #     np.array((-1, 0)),
        #     # right
        #     np.array((0, 1)),
        #     # down
        #     np.array((1, 0)),
        #     # left
        #     np.array((0, -1)),
        # ]
        # self._possible_next_states = [np.add(np.array([self._cX, self._cY]), dir) for dir in DIR_TO_VEC]
        # self._possible_next_states = [c if (np.all(c >= 0) and
        #                                     (c[0] < self._height and
        #                                      c[1] < self._width) and
        #                                     self._mdp[c[0], c[1]] != -1)
        #                               else np.array([self._cX, self._cY])
        #                               for c in self._possible_next_states]
        #
        # next_state = np.add(np.array([self._cX, self._cY]), DIR_TO_VEC[action])
        # next_state = next_state if (np.all(next_state >= 0) and
        #                                     (next_state[0] < self._height and
        #                                      next_state[1] < self._width) and
        #                                     self._mdp[next_state[0], next_state[1]] != -1) else np.array([self._cX, self._cY])
        #
        # if self._stochastic:
        #     slip_prob = [self._slip_prob, 1 - self._slip_prob]
        #     random_move = self._rng.choice(a=np.arange(4),
        #                      p=[1/4] * 4)
        #     next_states = [self._possible_next_states[random_move], next_state]
        #     next_state = self._rng.choice([0, 1], p=slip_prob)
        #     next_state = next_states[next_state]
        #
        # return next_state[0], next_state[1]

    def reset(self):
        """Returns the first `TimeStep` of a new episode."""
        self._reset_next_step = False
        sPos = self._rng.choice(range(self._nS),
                               p=self._d)
        self._cX, self._cY = self._get_state_coords(sPos)
        # if self._random_restarts:
        #     valid = False
        #     while not valid:
        #         self._sX = self._rng.randint(self._height)
        #         self._sY = self._rng.randint(self._width)
        #         if self._mdp[self._sX][self._sY] != -1 and \
        #                 (not ((self._sX, self.sY) in self._g)):
        #             valid = True
        # self._cX = self._sX
        # self._cY = self._sY
        return dm_env.restart(self._observation())

    def step(self, action):
        """Updates the environment according to the action."""
        if self._reset_next_step:
            return self.reset()

        nX, nY = self._get_next_state((self._cX, self._cY), action)
        reward = self._get_next_reward(nX, nY)
        self._cX = nX
        self._cY = nY

        if self._is_terminal():
            self._reset_next_step = True
            # self._cX, self._cY = self._sX, self._sY
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


    def _fill_P_R(self):
        self._P = np.zeros((self._nA, self._nS, self._nS), dtype=np.float)
        self._P_absorbing = np.zeros((self._nA, self._nS, self._nS), dtype=np.float)
        self._R = np.zeros((self._nA, self._nS, self._nS), dtype=np.float)
        self._d = np.zeros((self._nS), dtype=np.float)
        self._true_discount = np.ones((self._nS), dtype=np.float)
        self._index_matrix = np.zeros((self._height, self._width), dtype=np.int)

        for i in range(self._height):
            for j in range(self._width):
                self._index_matrix[i][j] = np.ravel_multi_index((i, j), (self._height, self._width))

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

        for i in range(self._height):
            for j in range(self._width):
                # if self._mdp[i][j] != -1 and not ((i, j) in self._g):
                if (i, j) in self._starting_states:
                # if i == self._sX and j == self._sY:
                    self._d[self._index_matrix[i][j]] = 1
                if (i, j) in self._g:
                    self._true_discount[self._index_matrix[i][j]] = 0

        self._d = np.divide(self._d, np.sum(self._d),
                            out=np.zeros_like(self._d),
                            where=np.sum(self._d) != 0)


        for i in range(self._height):
            for j in range(self._width):
                for k in range(self._nA):
                    fwd_pos = np.array([i, j]) + DIR_TO_VEC[k]
                    fwd_i, fwd_j = fwd_pos
                    if self._mdp[i][j] != -1:
                        if not ((i, j) in self._g):
                            if fwd_i >= 0 and fwd_i < self._height and\
                                fwd_j >= 0 and fwd_j < self._width and self._mdp[fwd_i][fwd_j] != -1:
                                if self._stochastic:
                                    slip_prob = [self._slip_prob, 1 - self._slip_prob]
                                else:
                                    slip_prob = [0, 1]
                                # prob of transitioning to the next state
                                self._P[k][self._index_matrix[i][j]][self._index_matrix[fwd_i][fwd_j]] = slip_prob[1]
                                self._P_absorbing[k][self._index_matrix[i][j]][self._index_matrix[fwd_i][fwd_j]] = slip_prob[1]

                                # reward incurred if transitioning to the next state
                                self._R[k][self._index_matrix[i][j]][self._index_matrix[fwd_i][fwd_j]] = \
                                                self._max_reward * 0.1 if (fwd_i, fwd_j) in self._g else 0

                                # prob of slipping and staying in the current state
                                self._P[k][self._index_matrix[i][j]][self._index_matrix[i][j]] = slip_prob[0]
                                self._P_absorbing[k][self._index_matrix[i][j]][self._index_matrix[i][j]] = slip_prob[0]

                                # reward incurred in the current state
                                self._R[k][self._index_matrix[i][j]][self._index_matrix[i][j]] = \
                                        self._max_reward * 0.1 if (i, j) in self._g else 0 # self._get_next_reward(i, j)
                            else:
                                # automatically staying in the current state because you bumped into the edge of the world
                                self._P[k][self._index_matrix[i][j]][self._index_matrix[i][j]] = 1
                                self._P_absorbing[k][self._index_matrix[i][j]][self._index_matrix[i][j]] = 1
                                self._R[k][self._index_matrix[i][j]][self._index_matrix[i][j]] = \
                                    self._max_reward * 0.1 if (i, j) in self._g else 0 #self._get_next_reward(i, j)
                        else:
                            # the modified ergodic MDP resets to the starting state with probability 1 if at the goal
                            for si in range(self._height):
                                for sj in range(self._width):
                                    self._P[k][self._index_matrix[i][j]][self._index_matrix[si][sj]] = \
                                        self._d[self._index_matrix[si][sj]]
                            # the original absorbing MDP stays at the goal forever
                            self._P_absorbing[k][self._index_matrix[i][j]][self._index_matrix[i][j]] = 1
                            # reward upon transitioning at the goal forever, or reseting to the starting state is 0
                            self._R[k][self._index_matrix[i][j]][self._index_matrix[i][j]] = 0


    def _get_dynamics(self, feature_coder=None):
        if self._P is None or self._R is None or self._P_absorbing is None or self._d is None:
            self._fill_P_R()

        return self._P, self._P_absorbing, self._R, self._d, self._true_discount

    def reshape_v(self, v):
        return np.reshape(v, (self._height, self._width))

    def reshape_q(self, v):
        return np.reshape(v, (self._height, self._width, self._nA))

    def reshape_pi(self, pi):
        return np.reshape(pi, (self._height, self._width, self._nA))

    def get_all_states(self):
        states = []
        for i in range(self._height):
            for j in range(self._width):
                index = self._get_state_index(i, j)
                # index = np.ravel_multi_index((i, j), (self._height, self._width))
                if self._obs_type == "onehot":
                    onehotstate = np.eye(self._nS)[index]
                    states.append(onehotstate)
                else:
                    states.append(index)
        return states

if __name__ == "__main__":
    nrng = np.random.RandomState(0)
    nS = None
    nA = 4
    discount = 0.99
    mdp_filename = "../../mdps/maze_48_open_goal.mdp"
    env = MicroWorld(path=mdp_filename, stochastic=False,
                        rng=nrng, env_size=48)
    nS = env._nS
    nA = 4

    plot_grid(env, logs=str(os.environ['LOGS']), env_type="discrete")
    mdp_solver = MdpSolver(env, nS, nA, discount)
    v = mdp_solver.get_optimal_v()
    # v = env.reshape_v(v)
    # pi = mdp_solver.get_optimal_policy()
    # policy = lambda x, nrng: np.argmax(pi[x])

    # mdp_solver = MdpSolver(env, nS, nA, discount)
    # # policy = mdp_solver.get_optimal_policy()
    # v = mdp_solver.get_optimal_v()
    v = env.reshape_v(v)
    plot_v(env, v, str(os.environ['LOGS']), env_type="discrete")
    # plot_policy(env, env.reshape_pi(env._pi), str((os.environ['LOGS'])),
    #             env_type="continuous")
    pi = np.full((env._nS, env._nA), 1 / env._nA)
    # eta_pi = mdp_solver.get_eta_pi(pi)

    plot_error(env, v, str(os.environ['LOGS']), env_type="continuous",
               eta_pi=env._d,
               filename="v_eta.png")


    # plot_eta_pi(env, env.reshape_v(eta_pi))