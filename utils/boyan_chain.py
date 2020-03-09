import dm_env
from dm_env import specs
import numpy as np
import scipy

class BoyanChain(dm_env.Environment):
    """
    Boyan Chain example. All states form a chain with ongoing arcs and a
    single terminal state. From a state one transition with equal probability
    to either the direct or the second successor state. All transitions have
    a reward of -3 except going from second to last to last state (-2)
    """

    def __init__(self, rng=None, obs_type="tabular", nS = 14, nF=4):
        self._nS = nS
        self._nF = nF
        self._rng = rng
        self._obs_type = obs_type
        self._states = np.arange(nS)
        self._nA = 1
        self._actions = np.arange(1)
        self._d0 = np.zeros(nS)
        self._d0[0] = 1
        self._r = np.ones((nS, 1, nS)) * (-3)
        self._r[nS - 2, :, nS - 1] = -2
        self._r[nS - 1:, :, :] = 0
        self._P = np.zeros((nS, 1, nS))
        self._P[-1, :, -1] = 1
        self._P[nS - 2, :, nS - 1] = 1
        for s in np.arange(nS - 2):
            self._P[s, :, s + 1] = 0.5
            self._P[s, :, s + 2] = 0.5

        self._terminal_trans = nS
        self.Phi = {}
        self._start_state = 0

        # start distribution testing
        self._d0 = np.asanyarray(self._d0)
        self._P = np.asanyarray(self._P)

        # extract valid actions and terminal state information
        sums_s = np.sum(self._P, axis=2)
        self._valid_actions = np.abs(sums_s - 1) < 0.0001
        self._s_terminal = np.asarray([np.all(self._P[s, :, s] == 1)
                                      for s in self._states])

    def reset(self):
        """Returns the first `TimeStep` of a new episode."""
        self._reset_next_step = False
        self._state = self._start_state
        return dm_env.restart(self._observation())

    def _get_next_state(self, action):
        next_state = self._rng.choice(np.arange(self._nS), p=self._P[self._state, action])

        return next_state

    def _get_next_reward(self, a, next_state):
        reward = self._r[self._state, a, next_state]
        return reward

    def _is_terminal(self):
        if self._state == self._nS - 1:
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
        elif self._obs_type == "spikes":
            a = (self._nS - 1.) / (self._nF - 1)
            r = 1 - abs((self._state + 1 - np.linspace(1, self._nS, self._nF)) / a)
            r[r < 0] = 0
            return r

    def _get_dynamics(self):
        return self._P, self._P, self._r

    def reshape_v(self, v):
        return np.reshape(v, (self._nS))

    def reshape_pi(self, pi):
        return np.reshape(pi, (self._nS, self._nA))

    def get_all_states(self):
        return np.arange(0, self._nS)