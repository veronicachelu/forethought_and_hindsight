import dm_env
import numpy as np
from dm_env import specs


class ActionNoisyBandit(dm_env.Environment):
    def __init__(self, rng=None, obs_type="tabular", nS=1, nA = 5,
                 noise_scale=1.5):
        self._P = None
        self._R = None
        self._stochastic = False
        self._nS = nS
        self._start_state = 0
        self._rng = rng
        self._nA = nA
        self._noise_scale = noise_scale
        action_mask = self._rng.choice(
            range(self._nA), size=self._nA, replace=False)
        self._rewards = np.linspace(0, 1, self._nA)[action_mask]
        self._total_regret = 0.
        self._optimal_return = 1.
        self._obs_type = obs_type

        self._reset_next_step = True

    def reset(self):
        """Returns the first `TimeStep` of a new episode."""
        self._reset_next_step = False
        self._state = self._start_state
        return dm_env.restart(self._observation())

    def _get_next_reward(self, action):
        reward = self._rewards[action]
        self._total_regret += self._optimal_return - reward

        reward = self._add_reward_noise(reward)

        return reward

    def _add_reward_noise(self, reward):
        reward = reward + self._noise_scale * self._rng.randn()
        return reward

    def _is_terminal(self):
        return True

    def step(self, action):
        """Updates the environment according to the action."""
        if self._reset_next_step:
            return self.reset()

        reward = self._get_next_reward(action)
        self._state = self._start_state

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

if __name__ == "__main__":
    nrng = np.random.RandomState(0)
    nS = 1
    nA = 2
    discount = 0.9
    env = ActionNoisyBandit(rng=nrng, obs_type="tabular", nA=2)
