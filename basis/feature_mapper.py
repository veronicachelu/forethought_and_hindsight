# Python imports
import sys
import basis.tile as tile
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
            self.get_features = self.get_rbf_features

    def get_tile_features(self, observations):
        states = []
        for o in observations:
            encoded_obs = tile.tile_encode(o, self._tilings)[0]
            index = np.ravel_multi_index(encoded_obs, self._bins)
            onehotstate = np.eye(np.prod(self._bins))[index]
            states.append(onehotstate)
        return np.array(states)

    def get_rbf_features(self, observations):
        # Perform feature bucketing.
        new_features = [np.vectorize(lambda x: -(x) ** 2)(o)
                        for o in observations]

        return np.array(new_features)


