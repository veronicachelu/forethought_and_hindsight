import numpy as np
import os
from copy import deepcopy

class MdpSolver:
    def __init__(self, env,
                 nS, nA, discount, feature_coder=None):
        self._p, self._p_absorbing, self._r, self._d = env._get_dynamics(feature_coder)
        if feature_coder is not None:
            self._nS = np.prod(feature_coder["num_tiles"]) * feature_coder["num_tilings"]
        else:
            self._nS = nS
        self._nA = nA
        self._env = env
        self._v = None
        self._discount = discount
        self._pi = None
        self._eta_pi = None
        self._theta = 1e-4
        self._pi = np.full((self._nS, self._nA), 1 / self._nA)
        self._assigned_pi = False

        mdp_root_path = os.path.split(os.path.split(env._path)[0])[0]
        baseline = os.path.basename(env._path)
        mdp_name = os.path.splitext(baseline)[0]
        policy_dir = os.path.join(mdp_root_path, "policies")
        if not os.path.exists(policy_dir):
            os.makedirs(policy_dir)
        self.policy_path = os.path.join(policy_dir, "{}_{}.npy".format(mdp_name,
                                                                       "stochastic"
                                                                       if env._stochastic
                                                                       else "deterministic"))

        if os.path.exists(self.policy_path):
            self._pi = np.load(self.policy_path, allow_pickle=True)#[()]
            self._assigned_pi = True
            self._solve_mdp()

    def _solve_mdp(self):
        ppi = np.einsum('kij, ik->ij', self._p_absorbing, self._pi)
        rpi = np.einsum('kij, kij, ik->i', self._r, self._p_absorbing, self._pi)
        self._v = np.linalg.solve(np.eye(self._r.shape[1]) - self._discount * ppi, rpi)

    def _improve_policy(self):
        done = True
        for s in range(self._nS):
            old_action = np.argmax(self._pi[s])
            # action_lookahead = np.vectorize(lambda a: np.sum(
            #     np.vectorize(lambda s_prime: self._p[a][s][s_prime] * (self._r[a][s][s_prime] + self._discount * self._v[s_prime]))
            #                        (range(self._nS))))
            action_lookahead = np.vectorize(lambda a: np.sum(
                np.vectorize(lambda s_prime: self._p_absorbing[a][s][s_prime] * (
                self._r[a][s][s_prime] + self._discount * self._v[s_prime]))
                (range(self._nS))))
            max_a = np.argmax(action_lookahead(range(self._nA)))
            self._pi[s] = np.eye(self._nA)[max_a]
            if old_action != max_a:
                if np.abs(self._v[old_action] - self._v[max_a]) > self._theta:
                    done = False
        return done

    def _policy_iteration(self):
        self._pi = np.full((self._nS, self._nA), 1 / self._nA)
        done = False
        while not done:
            self._solve_mdp()
            done = self._improve_policy()

        self._solve_mdp()

    def get_eta_pi(self, pi):
        if self._eta_pi is None:
            ppi = np.einsum('kij, ik->ij', self._p, pi)
            A = np.eye(self._nS) - ppi
            A = np.vstack((A.T, np.ones(self._nS)))
            b = np.matrix([0] * self._nS + [1]).T
            eta_pi = np.linalg.lstsq(A, b)[0]
            self._eta_pi = np.array(eta_pi.T)[0]

        return self._eta_pi

    def get_optimal_policy(self):
        if self._pi is None or self._assigned_pi is False:
            self._policy_iteration()
            np.save(self.policy_path, self._pi)
            self._assigned_pi = True

        return self._pi

    def get_optimal_v(self):
        if self._v is None:
            self._solve_mdp()

        return self._v

    # def get_value_for_state(self, state):
    #     return self._v[np.argmax(state)]

    def get_value_for_state(self, agent, environment, timestep):
        return self._run_agent_from_state(agent, environment, timestep,
                                          20, 100000)
        # return self._v(state)

    def _run_agent_from_state(self, agent, environment, timestep,
                              num_trajectories, max_len):
        traj_rewards = []
        traj_steps = []
        for _ in np.arange(start=0, stop=num_trajectories):
            traj_env = deepcopy(environment)
            traj_reward = 0
            for t in range(max_len):

                action = agent.policy(timestep, eval=True)
                new_timestep = traj_env.step(action)

                traj_reward += new_timestep.reward

                if new_timestep.last():
                    break
                timestep = new_timestep
            traj_rewards.append(traj_reward)
            traj_steps.append(t)

        # print("traj {}, {}".format(np.mean(traj_rewards), np.mean(traj_steps)))
        return np.mean(traj_rewards)
