import dm_env
import numpy as np
from dm_env import specs

from utils.mdp_solvers.solve_chain import ChainSolver
import itertools

class Bipartite(dm_env.Environment):
    def __init__(self, rng=None, obs_type="tabular", nS = (5,5,5,5,5)):
        self._P = None
        self._R = None
        self._stochastic = False
        self._nS = np.sum(nS)
        self._nL = len(nS)
        self._start_states = np.arange(nS[0])
        self._end_states = np.arange(self._nS - nS[-1], self._nS)
        self._mid_layers = []
        self._rng = rng
        self._nA = 1
        self._slip_prob = 0
        self._obs_type = obs_type
        self._pi = np.full((self._nS, self._nA), 1 / self._nA)

        if len(nS) > 2:
            before = nS[0]
            for l in range(1, len(nS)-1):
                self._mid_layers.append(np.arange(before, before + nS[l]))
                before += nS[l]
        self._rewards = np.full((self._nS, self._nS), 0, dtype=np.float)
        if len(nS) > 2:
            for state in self._mid_layers[-1]:
                self._rewards[state, self._end_states] = \
                    self._rng.normal(loc=10, scale=10.0, size=len(self._end_states))
        else:
            for start_state in self._start_states:
                self._rewards[start_state, self._end_states] = \
                    self._rng.normal(loc=10, scale=10.0, size=len(self._end_states))
        self._starting_distribution = self._rng.uniform(size=len(self._start_states))
        self._starting_distribution /= np.sum(self._starting_distribution)
        self._fill_P_R()
        self._reset_next_step = True

    def reset(self):
        """Returns the first `TimeStep` of a new episode."""
        self._reset_next_step = False
        self._state = self._rng.choice(self._start_states,
                                          p=self._starting_distribution)
        return dm_env.restart(self._observation())

    def _get_next_state(self, state, action):
        next_state = self._rng.choice(np.arange(self._nS), p=self._P[action][state])
        return next_state

    def _get_next_reward(self, state, next_state):
        reward = self._rewards[state, next_state]
        return reward

    def _is_terminal(self):
        if self._state in self._end_states:
            return True
        return False

    def step(self, action):
        """Updates the environment according to the action."""
        if self._reset_next_step:
            return self.reset()

        next_state = self._get_next_state(self._state, action)
        reward = self._get_next_reward(self._state, next_state)
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
                if s in self._start_states:
                    if len(self._mid_layers) == 0:
                        little_p = self._rng.uniform(size=len(self._end_states))
                        little_p /= np.sum(little_p)
                        for i, fwd_s in enumerate(self._end_states):
                            self._P[k][s][fwd_s] = little_p[i]
                            self._P_absorbing[k][s][fwd_s] = little_p[i]
                            self._R[k][s][fwd_s] = self._get_next_reward(s, fwd_s)
                    else:
                        little_p = self._rng.uniform(size=len(self._mid_layers[0]))
                        little_p /= np.sum(little_p)
                        for i, fwd_s in enumerate(self._mid_layers[0]):
                            self._P[k][s][fwd_s] = little_p[i]
                            self._P_absorbing[k][s][fwd_s] = little_p[i]
                            self._R[k][s][fwd_s] = self._get_next_reward(s, fwd_s)
                elif s in self._end_states:
                    for i, fwd_s in enumerate(self._start_states):
                        self._P[k][s][fwd_s] = self._starting_distribution[i]
                        self._R[k][s][fwd_s] = self._get_next_reward(s, fwd_s)
                    self._P_absorbing[k][s][s] = 1
                else:
                    for layer_idx, layer in enumerate(self._mid_layers):
                        if s in layer and layer_idx == len(self._mid_layers) - 1:
                            little_p = self._rng.uniform(size=len(self._end_states))
                            little_p /= np.sum(little_p)
                            for i, fwd_s in enumerate(self._end_states):
                                self._P[k][s][fwd_s] = little_p[i]
                                self._P_absorbing[k][s][fwd_s] = little_p[i]
                                self._R[k][s][fwd_s] = self._get_next_reward(s, fwd_s)
                        elif s in layer:
                            little_p = self._rng.uniform(size=len(self._mid_layers[layer_idx + 1]))
                            little_p /= np.sum(little_p)
                            for i, fwd_s in enumerate(self._mid_layers[layer_idx + 1]):
                                self._P[k][s][fwd_s] = little_p[i]
                                self._P_absorbing[k][s][fwd_s] = little_p[i]
                                self._R[k][s][fwd_s] = self._get_next_reward(s, fwd_s)

    def _get_dynamics(self):
        if self._P is None or self._R is None or self._P_absorbing is None:
            self._fill_P_R()

        return self._P, self._P_absorbing, self._R

    def reshape_v(self, v):
        return np.reshape(v, (self._nS))

    def reshape_pi(self, pi):
        return np.reshape(pi, (self._nS, self._nA))

    def get_all_states(self):
        states = []
        for s in self._start_states:
            self._state = s
            states.append(self._observation())
        return states

if __name__ == "__main__":
    nrng = np.random.RandomState(0)
    # nS = 10
    # nA = 1
    discount = 0.9
    env = Bipartite(rng=nrng, obs_type="tabular",
                 nS=(5, 1))
    mdp_solver = ChainSolver(env, 6, 1, discount)
    # policy = mdp_solver.get_optimal_policy()
    v = mdp_solver.get_optimal_v()
    # v = env.reshape_v(v)
    # plot_v(env, v, logs, env_type=FLAGS.env_type)
    # plot_policy(env, env.reshape_pi(policy), logs, env_type=FLAGS.env_type)
    eta_pi = mdp_solver.get_eta_pi(env._pi)
    # plot_eta_pi(env, env.reshape_v(eta_pi)