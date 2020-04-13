import dm_env
import numpy as np
from dm_env import specs

from utils.mdp_solvers.solve_chain import ChainSolver


class ActionLoop(dm_env.Environment):
    def __init__(self, rng=None, obs_type="tabular", nS = 50,
                 loop_prob=0):
        self._P = None
        self._R = None
        self._stochastic = False
        self._nS = nS
        self._start_state = 0
        self._end_states = [self._nS - 1]
        self._rng = rng
        self._nA = 2
        self._right_action = 0
        self._right_reward = 1
        self._slip_prob = 0
        self._obs_type = obs_type
        self._pi = np.full((self._nS, self._nA), 1 / self._nA)
        self._loop_prob = loop_prob
        self._reset_next_step = True

    def reset(self):
        """Returns the first `TimeStep` of a new episode."""
        self._reset_next_step = False
        self._state = self._start_state
        return dm_env.restart(self._observation())

    def _get_next_state(self, state, action):
        if action == self._right_action:
            next_state = state + 1
        else:
            next_state = self._start_state

        return next_state

    def _get_next_reward(self, state, action, next_state):
        if next_state == self._nS - 1 and action == self._right_action:
            reward = self._right_reward
        else:
            reward = 0

        return reward

    def _is_terminal(self, state, action, next_state):
        if state in self._end_states and action == self._right_action:
            return True
        return False

    def step(self, action):
        """Updates the environment according to the action."""
        if self._reset_next_step:
            return self.reset()

        next_state = self._get_next_state(self._state, action)
        reward = self._get_next_reward(self._state, action, next_state)
        isTerminal = self._is_terminal(self._state, action, next_state)
        self._state = next_state

        if isTerminal:
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
        elif self._obs_type == "inverted_features":
            return specs.BoundedArray(shape=(self._nS,), dtype=np.int32,
                                      name="state", minimum=0, maximum=1)
        elif self._obs_type == "dependent_features":
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
        elif self._obs_type == "dependent_features":
            return self._dependent_features[self._state]
        elif self._obs_type == "inverted_features":
            return self._inverted_features[self._state]

    def _fill_P_R(self):
        self._P = np.zeros((self._nA, self._nS, self._nS), dtype=np.float)
        self._P_absorbing = np.zeros((self._nA, self._nS, self._nS), dtype=np.float)
        self._R = np.zeros((self._nA, self._nS, self._nS), dtype=np.float)

        for s in range(self._nS):
            for k in range(self._nA):
                if not (s in self._end_states):
                    fwd_s = s + 1
                    if s == self._nS - 2:
                        self._P[k][s][fwd_s] = 1 - self._loop_prob
                        self._P[k][s][s] = self._loop_prob
                        self._P_absorbing[k][s][fwd_s] = 1 - self._loop_prob
                        self._P_absorbing[k][s][s] = self._loop_prob
                    else:
                        self._P[k][s][fwd_s] = 1
                        self._P_absorbing[k][s][fwd_s] = 1

                    self._R[k][s][fwd_s] = self._get_next_reward(fwd_s)
                    # self._P[k][s][s] = 0
                    # self._P_absorbing[k][s][s] = 0
                    # self._R[k][s][s] = self._get_next_reward(s)
                else:
                    self._P[k][s][self._start_state] = 1
                    self._P_absorbing[k][s][s] = 1
                    # self._R[k][s][s] = 0
        pass

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
    nS = 5
    nA = 2
    discount = 0.95
    env = ActionLoop(rng=nrng, obs_type="tabular",
                     nS=nS, loop_prob=0.1)
    mdp_solver = ChainSolver(env, nS, nA, discount)
    # policy = mdp_solver.get_optimal_policy()
    v = mdp_solver.get_optimal_v()
    v = env.reshape_v(v)
    # plot_v(env, v, logs, env_type=FLAGS.env_type)
    # plot_policy(env, env.reshape_pi(policy), logs, env_type=FLAGS.env_type)
    # eta_pi = mdp_solver.get_eta_pi(policy)
    # plot_eta_pi(env, env.reshape_v(eta_pi)