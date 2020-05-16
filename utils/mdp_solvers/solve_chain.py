import numpy as np
import os

class ChainSolver:
    def __init__(self, env,
                 nS, nA, discount):
        self._p, self._p_absorbing, self._r, self._d = env._get_dynamics()
        if isinstance(nS, tuple):
            nS = np.sum(nS)
        self._nS = nS
        self._nA = nA
        self._env = env
        self._v = None
        self._discount = discount
        self._pi = None
        self._eta_pi = None
        self._theta = 1e-8
        self._pi = np.full((self._nS, self._nA), 1 / self._nA)
        self._assigned_pi = False
        self._true_model = None

        # mdp_root_path = os.path.split(os.path.split(env._path)[0])[0]
        # baseline = os.path.basename(env._path)
        # mdp_name = os.path.splitext(baseline)[0]
        # policy_dir = os.path.join(mdp_root_path, "policies")
        # if not os.path.exists(policy_dir):
        #     os.makedirs(policy_dir)
        # self.policy_path = os.path.join(policy_dir, "{}_{}.npy".format(mdp_name,
        #                                                                "stochastic"
        #                                                                if env._stochastic
        #                                                                else "deterministic"))
        #
        # if os.path.exists(self.policy_path):
        #     self._pi = np.load(self.policy_path, allow_pickle=True)#[()]
        #     self._assigned_pi = True
        #     self._solve_mdp()

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
            # A = np.eye(self._nS) - self._discount * ppi
            A = np.vstack((A.T, np.ones(self._nS)))
            b = np.matrix([0] * self._nS + [1]).T
            # b = np.matrix(list(self._d) + [1]).T
            eta_pi = np.linalg.lstsq(A, b)[0]
            self._eta_pi = np.array(eta_pi.T)[0]

        return self._eta_pi

    def get_optimal_policy(self):
        if self._pi is None or self._assigned_pi is False:
            self._policy_iteration()
            # np.save(self.policy_path, self._pi)
            self._assigned_pi = True

        return self._pi

    def get_optimal_v(self):
        if self._v is None:
            self._solve_mdp()

        return self._v

    def get_true_model(self):
        if self._true_model is None:
            v = self.get_optimal_v()
            ppi = np.einsum('kij, ik->ij', self._p, self._pi)
            eta_pi = self.get_eta_pi(self._pi)
            stationary = np.diag(eta_pi)
            bw_model = np.matmul(np.matmul(np.linalg.inv(stationary), ppi.transpose()), stationary)
            self._true_model = (bw_model, ppi, np.mean(self._r, axis=0), v)

        return self._true_model