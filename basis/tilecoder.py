# Python imports
import sys
import numpy as np


class TileCoder(object):

    def __init__(self, ranges, num_tiles, num_tilings):
        self.ranges = ranges
        self.num_tiles = num_tiles
        self.num_tilings = num_tilings
        self.li_tile_size = [float(x[1] - x[0]) / (nt - 1) for x, nt in zip(ranges, num_tiles)]
        self.features_per_tiling = np.product(num_tiles)
        self.max_feature = self.features_per_tiling * self.num_tilings
        print("This Tiling produces features that range from 0 to {} ".format(self.max_feature - 1))

    def get_features(self, observations):
        # Clip input variables.
        clipped_state = self.clip_state_variables(observations)

        # Ensure range are set appropriately.s
        assert all([s <= r[1] for s, r in
                    zip(clipped_state, self.ranges)]), "state variable bigger than prespecified range ..."
        assert all([s >= r[0] for s, r in
                    zip(clipped_state, self.ranges)]), "state variable smaller than prespecified range ..."

        features = []
        for tiling in range(self.num_tilings):
            tiling_offset = [float(tiling) / (self.num_tilings * (nt - 1)) for nt in self.num_tiles]
            li_offset = [-x[0] + float(x[1] - x[0]) * to for x, to in zip(self.ranges, tiling_offset)]
            state_shifted = [s + o for s, o in zip(clipped_state, li_offset)]
            li_indices_in_tiling = [np.floor(float(ss) / lts) for ss, lts in zip(state_shifted, self.li_tile_size)]
            feature_index = np.sum(
                [li_indices_in_tiling[i] * np.product(self.num_tiles[0:i]) for i in range(len(self.num_tiles))])
            features.append(int(feature_index + tiling * self.features_per_tiling))

        vec = self.max_feature * [0]
        for f in features:
            vec[f] = 1
        return vec

    def clip_state_variables(self, observations):
        state_clipped = [np.clip(s, r[0], r[1]) for s, r in zip(observations, self.ranges)]
        return state_clipped


if __name__ == '__main__':
    # ranges = [[0, 1], [-1, 2], [0, 3]]
    ranges = [[0.0, 1.0], [0.0, 1.0]]
    tc = TileCoder(ranges, num_tiles=[4, 4], num_tilings=1)
    features = tc.get_features([0.5, 0.5])
    print(features)