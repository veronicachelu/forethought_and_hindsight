import numpy as np
env_config = {
    "class": "ObstacleWorld",
    "non_gridworld": False,
    "model_class": "linear",
    "feature_coder": {
        "type": "tile",
        "ranges": [[0.0, 0.0], [1.0, 1.0]],
        "num_tiles": [5, 5],
        "num_tilings": 1},
    "env_type": "continuous",
    "obs_type": "position",
    "policy_type": "greedy",
    "mdp_filename": "./continuous_mdps/obstacle.mdp",
    "env_size": None,
    "num_episodes": 100,
    "num_runs": 5,
    "stochastic": False,
    "nA": 4
}

volatile_agent_config = {
    "vanilla": {
        "planning_depth": [0],
        "replay_capacity": [0],
        "lr": np.linspace(0.01, 0.1, 10),
        "lr_p": [0],
        "lr_m": [0]
    },
    "bw": {
        "planning_depth": [1, 4, 8],
        "replay_capacity": [0],
        "lr": [0.01],
        "lr_p": [0.01],
        "lr_m": np.linspace(0.001, 0.07, 10),
    },
    "fw": {
        "planning_depth": [1, 4, 8],
        "replay_capacity": [0],
        "lr": [0.01],
        "lr_p": [0.01],
        "lr_m": np.linspace(0.001, 0.07, 10),
    },
    "fw_rnd": {
        "planning_depth": [1],
        "replay_capacity": [50],
        "lr": [0.01],
        "lr_p": [0.01],
        "lr_m": np.linspace(0.001, 0.07, 10),
    },
    # "ac_vanilla": {
    #     "planning_depth": [0],
    #     "replay_capacity": [0],
    #     "lr": np.linspace(0.01, 0.1, 10),
    #     "lr_p": [0],
    #     "lr_m": [0]
    # },
    # "vanilla_intr": {
    #     "planning_depth": [0],
    #     "replay_capacity": [0],
    #     "lr": np.linspace(0.01, 0.1, 10),
    #     "lr_p": [0],
    #     "lr_m": [0]
    # },
    # "latent_vanilla_intr": {
    #     "planning_depth": [0],
    #     "replay_capacity": [0],
    #     "lr": np.linspace(0.01, 0.1, 10),
    #     "lr_p": [0],
    #     "lr_m": [0]
    # },
    # "bw_intr": {
    #     "planning_depth": [1, 4, 8],
    #     "replay_capacity": [0],
    #     "lr": [0.08],
    #     "lr_p": [0.08],
    #     "lr_m": np.linspace(0.001, 0.08, 10),
    # },
    # "latent_bw_intr": {
    #     "planning_depth": [1, 4, 8],
    #     "replay_capacity": [0],
    #     "lr": [0.01],
    #     "lr_p": [0.01],
    #     "lr_m": np.linspace(0.001, 0.01, 10),
    # },

}