import dm_env
import numpy as np
from dm_env import specs

from utils.mdp_solvers.solve_chain import ChainSolver


class World3d(dm_env.Environment):
    def __init__(self, rng=None, obs_type="tabular", nS = 1000):
        self._P = None
        self._P_absrobing = None
        self._R = None
        self._stochastic = False
        self._rng = rng
        self._nS = nS
        self._nA = 6
        self._per_dim_states = 10
        self._obs_type = obs_type
        self._start_states = np.arange(nS)
        self._rewarded_states = self._rng.choice(self._start_states,
                                          p=[1/len(self._start_states) for _ in self._start_states],
                                         size=50, replace=False)
        self._rewards = np.zeros((self._nS))
        self._rewards[self._rewarded_states] = self._rng.normal(loc=0.0,
                              scale=1.0, size=(50))

        self._pi = np.full((self._nS, self._nA), 1 / self._nA)

        self._reset_next_step = True

        self._dir_to_vect = [
            # up
            np.array((-1, 0, 0)),
            # down
            np.array((1, 0, 0)),
            # fw
            np.array((0, -1, 0)),
            # right
            np.array((0, 0, 1)),
            # back
            np.array((0, 1, 0)),
            # left
            np.array((0, 0, -1)),
        ]

        self._fill_P_R()

    def reset(self):
        """Returns the first `TimeStep` of a new episode."""
        self._reset_next_step = False
        self._state = self._rng.choice(self._start_states,
                                          p=[1/len(self._start_states) for _ in self._start_states])
        return dm_env.restart(self._observation())

    def _get_next_state(self, state, action):
        next_state = self._rng.choice(np.arange(self._nS),
                                      p=self._P[action][state])

        return next_state

    def _get_next_reward(self, next_state):
         return self._rewards[next_state]

    def _is_terminal(self):
        return False

    def step(self, action):
        """Updates the environment according to the action."""
        if self._reset_next_step:
            return self.reset()

        next_state = self._get_next_state(self._state, action)
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

    def _get_state_index(self, coords):
        idx = np.ravel_multi_index(coords, (self._per_dim_states, self._per_dim_states, self._per_dim_states))
        return idx

    def _get_state_coords(self, idx):
        coords = np.unravel_index(idx, (self._per_dim_states, self._per_dim_states, self._per_dim_states))
        return coords

    def _fill_P_R(self):
        self._P = np.zeros((self._nA, self._nS, self._nS), dtype=np.float)
        self._P_absorbing = np.zeros((self._nA, self._nS, self._nS), dtype=np.float)
        self._R = np.zeros((self._nA, self._nS, self._nS), dtype=np.float)

        for s in range(self._nS):
            for k in range(self._nA):
                s_coords = self._get_state_coords(s)
                fwd_s = (s_coords + self._dir_to_vect[k]) % self._per_dim_states
                next_s = self._get_state_index(fwd_s)
                self._P[k][s][next_s] = 1
                self._P_absorbing[k][s][next_s] = 1
                self._R[k][s][next_s] = self._get_next_reward(next_s)

    def _get_dynamics(self):
        if self._P is None or self._R is None or self._P_absrobing is None:
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
    nS = 1000
    nA = 1
    discount = 0.95
    env = World3d(rng=nrng, obs_type="tabular",
                 nS=nS)
    mdp_solver = ChainSolver(env, nS, nA, discount)
    # policy = mdp_solver.get_optimal_policy()
    v = mdp_solver.get_optimal_v()
    v = env.reshape_v(v)
    # plot_v(env, v, logs, env_type=FLAGS.env_type)
    # plot_policy(env, env.reshape_pi(policy), logs, env_type=FLAGS.env_type)
    eta_pi = mdp_solver.get_eta_pi(mdp_solver._pi)
    # plot_eta_pi(env, env.reshape_v(eta_pi)