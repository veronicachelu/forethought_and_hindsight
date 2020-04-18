import math
import numpy as np

class RBF(object):

    def get_features(self, observations):
        # Perform feature bucketing.
        new_features = []

        for o in observations:
            new_features_per_o = []
            for i, feat in enumerate(o):
                new_feat = math.exp(-(feat)**2)
                new_features_per_o.append(new_feat)
            new_features.append(new_features_per_o)

        return np.array(new_features)