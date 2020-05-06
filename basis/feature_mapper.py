# Python imports
import sys
import basis.tile as tile
import basis.rbf as rbf
import numpy as np
import math

class FeatureMapper(object):
    def __init__(self, feature_coder):
        if feature_coder["type"] == "tile":
            self._tilings = [tile.create_tiling_grid(feature_coder["ranges"][0], feature_coder["ranges"][1],
                                                bins=(feature_coder["num_tiles"][0], feature_coder["num_tiles"][0]),
                                                offsets=(0.0, 0.0))]
            self._bins = (feature_coder["num_tiles"][0], feature_coder["num_tiles"][0])
            self.get_features = self.get_tile_features

        elif feature_coder["type"] == "rbf":
            self._rbf = rbf.RBF(low=feature_coder["ranges"][0],
                               high=feature_coder["ranges"][1],
                               num_centers=(feature_coder["num_centers"][0], feature_coder["num_centers"][0]),
                               offsets=(0.0, 0.0))

            self.get_features = self.get_rbf_features

            if feature_coder["noise"]:
                self.get_features = self.get_noisy_rbf_features
                self._noise_dim = feature_coder["noise_dim"]

    def get_tile_features(self, observations):
        states = []
        for o in observations:
            encoded_obs = tile.tile_encode(o, self._tilings)[0]
            index = np.ravel_multi_index(encoded_obs, self._bins)
            onehotstate = np.eye(np.prod(self._bins))[index]
            states.append(onehotstate)
        return np.array(states)

    def get_rbf_features(self, observations, nrng):
        features = self._rbf.get_features(observations)

        return features

    def get_noisy_rbf_features(self, observations, nrng):
        features = self._rbf.get_features(observations)

        batch_size = features.shape[0]
        noise = nrng.normal(loc=0.0,
                              scale=1.0, size=(batch_size, self._noise_dim))

        features = np.concatenate([features, noise], axis=-1)
        return features



