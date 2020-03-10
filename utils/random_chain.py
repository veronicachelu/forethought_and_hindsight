import dm_env
from dm_env import specs
import numpy as np
import scipy


class RandomChain(dm_env.Environment):
    def __init__(self, rng=None, obs_type="tabular", nS = 5,
                 left_reward=-1, right_reward=1):
        self._P = None
        self._R = None
        self._stochastic = False
        self._nS = nS
        self._start_state = nS // 2
        self._end_states = [0, self._nS - 1]
        self._rng = rng
        self._nA = 2
        self._left_reward = left_reward
        self._right_reward = right_reward
        self._slip_prob = 0
        self._obs_type = obs_type

        self._reset_next_step = True
        self._true_v = np.linspace(left_reward * nS, right_reward * nS, nS) / nS
        self._dependent_features = [[1, 0, 0],
                          [1/np.sqrt(2), 1/np.sqrt(2), 0],
                          [1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)],
                          [0, 1/np.sqrt(2), 1/np.sqrt(2)],
                          [0, 0, 1]]
        self._inverse_features = [[0, 1/2, 1/2, 1/2, 1/2],
                                  [1/2, 0, 1/2, 1/2, 1/2],
                                  [1/2, 1/2, 0, 1/2, 1/2],
                                  [1/2, 1/2, 1/2, 0, 1/2],
                                  [1/2, 1/2, 1/2, 1/2, 0]]
        # np.arange(left_reward * nS,
        #                          right_reward * (nS+1),
        #                          2) / nS
        # self._true_v[0] = self._true_v[-1] = 0

    def reset(self):
        """Returns the first `TimeStep` of a new episode."""
        self._reset_next_step = False
        self._state = self._start_state
        return dm_env.restart(self._observation())

    def _get_next_state(self, action):
        if action == 1:
            next_state = self._state + 1
        else:
            next_state = self._state - 1

        return next_state

    def _get_next_reward(self, next_state):
        if next_state == 0:
            reward = self._left_reward
        elif next_state == self._nS - 1:
            reward = self._right_reward
        else:
            reward = 0

        return reward

    def _is_terminal(self):
        if self._state in self._end_states:
            return True
        return False

    def step(self, action):
        """Updates the environment according to the action."""
        if self._reset_next_step:
            return self.reset()

        next_state = self._get_next_state(action)
        reward = self._get_next_reward(next_state)
        self._state = next_state

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

    def action_spec(self):
        return specs.DiscreteArray(
            dtype=int, num_values=self._nA, name="action")

    def _observation(self):
        if self._obs_type == "tabular":
            return self._state
        elif self._obs_type == "onehot":
            return np.eye(self._nS)[self._state]
        elif self._obs_type == "dependent_features":
            return self._features[self.state]

    def _fill_P_R(self):
        self._P = np.zeros((self._nA, self._nS + 1, self._nS + 1), dtype=np.int)
        self._P_absorbing = np.zeros((self._nA, self._nS + 1, self._nS + 1), dtype=np.int)
        self._R = np.zeros((self._nA, self._nS + 1, self._nS + 1), dtype=np.int)

        DIR_TO_VEC = [
            # left
            -1,
            # right
            +1
        ]
        for s in range(self._nS):
            for k in range(self._nA):
                fwd_s = s + DIR_TO_VEC[k]
                if not (s in self._end_states):
                    if self._stochastic:
                        slip_prob = [self._slip_prob, 1 - self._slip_prob]
                    else:
                        slip_prob = [0, 1]
                    self._P[k][s][fwd_s] = slip_prob[1]
                    self._P_absorbing[k][s][fwd_s] = slip_prob[1]
                    self._R[k][s][fwd_s] = self._get_next_reward(fwd_s)
                    self._P[k][s][s] = slip_prob[0]
                    self._P_absorbing[k][s][s] = slip_prob[0]
                    self._R[k][s][s] = self._get_next_reward(s)
                else:
                    self._P[k][s][self._start_state] = 0
                    self._P_absorbing[k][s][s] = 1
                    self._R[k][s][s] = 0

    def _get_dynamics(self):
        if self._P == None or self._R == None or self._P_absrobing == None:
            self._fill_P_R()

        return self._P, self._P_absorbing, self._R

    def reshape_v(self, v):
        return np.reshape(v, (self._nS))

    def reshape_pi(self, pi):
        return np.reshape(pi, (self._nS, self._nA))

    def get_all_states(self):
        states = []
        for s in np.arange(0, self._nS):
            self._state = s
            states.append(self._observation())
        return states