import math
import numpy as np
import functools
import itertools

class RBF(object):

    def __init__(self, low, high, num_centers, offsets):
        centers = [np.linspace(low[dim], high[dim], num_centers[dim] + 2, endpoint=False)[1:-1] + offsets[dim]
                         for dim in range(len(num_centers))]
        self._centers = np.array(list(itertools.product(*centers)))

        def rbf_fun(obs, rbf_center):
            rbf = np.exp(-(1/2) *
                            np.sum((obs - rbf_center)**2, axis=-1, keepdims=True))

            return rbf

        self._rbf_fun = rbf_fun

    def get_features(self, observations):
        batch_features = []
        for o in observations:
            features = np.vectorize(self._rbf_fun,
                                signature='(obs),(rbf_center)->(feat)',
                                otypes=[float])([o], self._centers)
            batch_features.append(np.squeeze(features, axis=-1))
        return np.array(batch_features)
        # Perform feature bucketing.
        # new_features = []
        #
        # for o in observations:
        #     new_features_per_o = []
        #     for i, feat in enumerate(o):
        #         new_feat = math.exp(-(feat)**2)
        #         new_features_per_o.append(new_feat)
        #     new_features.append(new_features_per_o)
        #
        # return np.array(new_features)