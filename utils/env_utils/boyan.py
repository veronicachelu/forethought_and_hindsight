import dm_env
import numpy as np
from dm_env import specs

from utils.mdp_solvers.solve_chain import ChainSolver


class Boyan(dm_env.Environment):
    def __init__(self, rng=None, obs_type="tabular", nS = 14, nF=4):
        self._P = None
        self._R = None
        self._stochastic = False
        self._nS = nS
        self._start_state = 0
        self._end_state = self._nS - 1
        self._last_state = self._nS - 2
        self._second_to_last = self._nS - 3
        self._nF = nF
        self._rng = rng
        self._nA = 1
        self._obs_type = obs_type
        self._pi = np.full((self._nS, self._nA), 1 / self._nA)

        self._reset_next_step = True


    def reset(self):
        """Returns the first `TimeStep` of a new episode."""
        self._reset_next_step = False
        self._state = self._start_state
        return dm_env.restart(self._observation())

    def _get_next_state(self, state, action):
        if state == self._nS - 3 or state == self._nS - 2:
            next_state = state + 1
        else:
            state_mask = self._rng.choice(range(2),
                                          p=[0.5, 0.5])
            next_state = state + state_mask + 1

        return next_state

    def _get_next_reward(self, state, action, next_state):
        if state == self._second_to_last:
            reward = -2.0
        elif state == self._last_state or state == self._end_state:
            reward = 0.0
        else:
            reward = -3.0

        return reward

    def _is_terminal(self):
        if self._state == self._end_state:
            return True
        return False

    def step(self, action):
        """Updates the environment according to the action."""
        if self._reset_next_step:
            return self.reset()

        next_state = self._get_next_state(self._state, action)
        reward = self._get_next_reward(self._state, action, next_state)
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
        elif self._obs_type == "spikes":
            return specs.BoundedArray(shape=(self._nF,), dtype=np.int32,
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


    def _fill_P_R(self):
        self._P = np.zeros((self._nA, self._nS, self._nS), dtype=np.float)
        self._P_absorbing = np.zeros((self._nA, self._nS, self._nS), dtype=np.float)
        self._R = np.zeros((self._nA, self._nS, self._nS), dtype=np.float)

        for s in range(self._nS):
            for k in range(self._nA):
                if s == self._second_to_last:
                    self._P[k][s][self._last_state] = 1
                    self._P_absorbing[k][s][self._last_state] = 1
                    self._R[k][s][self._last_state] = -2
                elif s == self._last_state:
                    self._P[k][s][self._end_state] = 1
                    self._P_absorbing[k][s][self._end_state] = 1
                    self._R[k][s][self._end_state] = 0
                elif s == self._end_state:
                    self._P[k][s][self._start_state] = 1
                    self._P_absorbing[k][s][s] = 1
                    self._R[k][s][s] = 0
                else:
                    self._P[k][s][s + 1] = 0.5
                    self._P[k][s][s + 2] = 0.5
                    self._P_absorbing[k][s][s + 1] = 0.5
                    self._P_absorbing[k][s][s + 2] = 0.5
                    self._R[k][s][s + 1] = -3
                    self._R[k][s][s + 2] = -3

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

if __name__ == "__main__":
    nrng = np.random.RandomState(0)
    nS = 14
    nF = 4
    nA = 1
    discount = 0.95
    env = Boyan(rng=nrng, obs_type="spikes",
               nS=nS, nF= nF)
    mdp_solver = ChainSolver(env, nS, nA, discount)
    # policy = mdp_solver.get_optimal_policy()
    v = mdp_solver.get_optimal_v()
    v = env.reshape_v(v)
    # plot_v(env, v, logs, env_type=FLAGS.env_type)
    # plot_policy(env, env.reshape_pi(policy), logs, env_type=FLAGS.env_type)
    # eta_pi = mdp_solver.get_eta_pi(policy)
    # plot_eta_pi(env, env.reshape_v(eta_pi)