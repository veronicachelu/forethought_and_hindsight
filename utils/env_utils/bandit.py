import dm_env
import numpy as np
from dm_env import specs

from utils.mdp_solvers.solve_chain import ChainSolver


class Bandit(dm_env.Environment):
    def __init__(self, rng=None, obs_type="tabular", nS=11,
                 noise_scale=1.5):
        self._P = None
        self._R = None
        self._nS = nS
        self._start_state = 0
        self._nA = (self._nS - 1) // 2
        self._end_states = range(self._nS)[self._nA:]
        self._interm_states = range(self._nS)[1:self._nA]
        self._rewards = np.linspace(0, 1, self._nA + 1)[1:]
        self._rng = rng
        self._noise_scale = noise_scale
        self._obs_type = obs_type
        self._slip_prob = 0

        self._reset_next_step = True

    def reset(self):
        """Returns the first `TimeStep` of a new episode."""
        self._reset_next_step = False
        self._state = self._start_state
        return dm_env.restart(self._observation())

    def _get_next_state(self, action):
        if self._state == self._start_state:
            next_state = self._state + action + 1
        else:
            next_state = self._state + self._nA

        return next_state

    def _get_next_reward(self, next_state):
        if next_state in self._end_states:
            reward = self._rewards[next_state - self._nA - 1]
            # reward = self._add_reward_noise(reward)
        else:
            reward = 0

        return reward

    def _add_reward_noise(self, reward):
        reward = reward + self._noise_scale * self._rng.randn()
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

    def action_spec(self):
        return specs.DiscreteArray(
            dtype=int, num_values=self._nA, name="action")

    def _observation(self):
        if self._obs_type == "tabular":
            return self._state

    def _fill_P_R(self):
        self._P = np.zeros((self._nA, self._nS, self._nS), dtype=np.float)
        self._P_absorbing = np.zeros((self._nA, self._nS, self._nS), dtype=np.float)
        self._R = np.zeros((self._nA, self._nS, self._nS), dtype=np.float)

        for s in range(self._nS):
            for k in range(self._nA):
                if s == self._start_state:
                    fwd_s = s + k + 1
                else:
                    fwd_s = s + self._nA
                if not (s in self._end_states):
                    self._P[k][s][fwd_s] = 1
                    self._P_absorbing[k][s][fwd_s] = 1
                    self._R[k][s][fwd_s] = self._get_next_reward(fwd_s)
                    self._P[k][s][s] = 0
                    self._P_absorbing[k][s][s] = 0
                    self._R[k][s][s] = self._get_next_reward(s)
                else:
                    self._P[k][s][self._start_state] = 1
                    self._P_absorbing[k][s][s] = 1
                    self._R[k][s][s] = 0

    def _get_dynamics(self):
        if self._P == None or self._R == None or self._P_absrobing == None:
            self._fill_P_R()

        return self._P, self._P_absorbing, self._R

if __name__ == "__main__":
    nrng = np.random.RandomState(0)
    discount = 0.95
    env = Bandit(rng=nrng)
    nS = env._nS
    nA = env._nA
    mdp_solver = ChainSolver(env, nS, nA, discount)
    v = mdp_solver.get_optimal_v()
    v = env.reshape_v(v)
