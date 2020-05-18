import numpy as np
env_config = {
    "class": "ObstacleWorld",
    "non_gridworld": False,
    "model_class": "linear",
    "feature_coder": {
        "type": "rbf",
        "ranges": [[0.0, 0.0], [10.0, 10.0]],
        "num_tiles": [10, 10],
        "num_centers": [4, 4],
        "num_tilings": 1,
        "noise": False,
        "noise_dim": 0,
    },
    "env_type": "continuous",
    "obs_type": "position",
    "policy_type": "continuous_random",
    "mdp_filename": "./continuous_mdps/obstacle.mdp",
    "env_size": None,
    "num_episodes": 50,
    # "total_steps": 10000,
    "control_num_episodes": 300,
    "num_runs": 1,
    "stochastic": False,
    "nA": 4
}

volatile_agent_config = {
    "vanilla": {
        "planning_depth": [0],
        "replay_capacity": [0],
        "lr": np.array([0.001, 0.005, 0.01, 0.05, 0.01]),
        "lr_p": [0],
        "lr_m": [0],
        "lr_ctrl": 0.4
    },
    "c_bw_PAML": {
        "planning_depth": [1],
        "replay_capacity": [0],
        "lr": [1.0],
        "lr_p": [1.0],
        "lr_m": np.array([0.001, 0.005, 0.01, 0.05]),
        "lr_ctrl": 0.4
    },
    "p_bw_PAML": {
        "planning_depth": [1],
        "replay_capacity": [0],
        "lr": [1.0],
        "lr_p": [1.0],
        "lr_m": np.array([0.001, 0.005, 0.01, 0.05]),
        "lr_ctrl": 0.4
    },
    "p_fw_PAML": {
        "planning_depth": [1],
        "replay_capacity": [0],
        "lr": [0.114],
        "lr_p": [0.114],
        "lr_m": np.array([0.001, 0.005, 0.01, 0.05]),
        "lr_ctrl": 0.4
    },
    "p_fw": {
        "planning_depth": [1],
        "replay_capacity": [0],
        "lr": [0.114],
        "lr_p": [0.114],
        "lr_m": np.array([0.001, 0.005, 0.01, 0.05]),
        "lr_ctrl": 0.4
    },
    "c_bw": {
        "planning_depth": [1],
        "replay_capacity": [0],
        "lr": [0.01],
        "lr_p": [0.01],
        "lr_m": np.array([0.001, 0.005, 0.01, 0.05]),
        "lr_ctrl": 0.4
    },
    "p_bw": {
        "planning_depth": [1],
        "replay_capacity": [0],
        "lr": [0.01],
        "lr_p": [0.01],
        "lr_m": np.array([0.001, 0.005, 0.01, 0.05]),
        "lr_ctrl": 0.4
    },
    "c_true_bw": {
        "planning_depth": [1],
        "replay_capacity": [0],
        "lr": [0.01],
        "lr_p": [0.01],
        "lr_m": np.array([0.001, 0.005, 0.01, 0.05]),
        "lr_ctrl": 0.4
    },
    "p_true_bw": {
        "planning_depth": [1],
        "replay_capacity": [0],
        "lr": [0.01],
        "lr_p": [0.01],
        "lr_m": np.array([0.001, 0.005, 0.01, 0.05]),
        "lr_ctrl": 0.4
    },
    # "bw_update": {
    #     "planning_depth": [1],
    #     "replay_capacity": [0],
    #     "lr": [1.0],
    #     "lr_p": [1.0],
    #     "lr_m": np.array([0.0001, 0.0005, 0.001, 0.005, 0.01]),
    #     "lr_ctrl": 0.4
    # },
    # "bw_mult_PAML": {
    #     "planning_depth": [1],
    #     "replay_capacity": [0],
    #     "lr": [1.0],
    #     "lr_p": [1.0],
    #     "lr_m": np.array([0.0001, 0.0005, 0.001, 0.005, 0.01]),
    #     "lr_ctrl": 0.4
    # },

    # "mult_fw": {
    #     "planning_depth": [1],
    #     "replay_capacity": [0],
    #     "lr": [0.114],
    #     "lr_p": [0.114],
    #     "lr_m": np.array([0.0001, 0.0005, 0.001, 0.005, 0.01]),
    #     "lr_ctrl": 0.4
    # },

    # "fw_mult_PAML": {
    #     "planning_depth": [1],
    #     "replay_capacity": [0],
    #     "lr": [0.114],
    #     "lr_p": [0.114],
    #     "lr_m": np.array([0.0001, 0.0005, 0.001, 0.005, 0.01]),
    #     "lr_ctrl": 0.4
    # },

    # "mult_bw": {
    #     "planning_depth": [1],
    #     "replay_capacity": [0],
    #     "lr": [0.01],
    #     "lr_p": [0.01],
    #     "lr_m": np.array([0.0001, 0.0005, 0.001, 0.005, 0.01]),
    #     "lr_ctrl": 0.4
    # },

    # "true_mult_bw": {
    #     "planning_depth": [1],
    #     "replay_capacity": [0],
    #     "lr": [0.114],
    #     "lr_p": [0.114],
    #      "lr_m": np.array([0.0001, 0.0005, 0.001, 0.005, 0.01]),
    #     "lr_ctrl": 0.4
    # },
    # "ac_vanilla": {
    #     "planning_depth": [0],
    #     "replay_capacity": [0],
    #     "lr": np.linspace(0.01, 0.1, 10),
    #     "lr_p": [0],
    #     "lr_m": [0]
    # },
    # "vanilla_PAML": {
    #     "planning_depth": [0],
    #     "replay_capacity": [0],
    #     "lr": np.linspace(0.01, 0.1, 10),
    #     "lr_p": [0],
    #     "lr_m": [0]
    # },
    # "latent_vanilla_PAML": {
    #     "planning_depth": [0],
    #     "replay_capacity": [0],
    #     "lr": np.linspace(0.01, 0.1, 10),
    #     "lr_p": [0],
    #     "lr_m": [0]
    # },
    # "bw_PAML": {
    #     "planning_depth": [1, 4, 8],
    #     "replay_capacity": [0],
    #     "lr": [0.08],
    #     "lr_p": [0.08],
    #     "lr_m": np.linspace(0.001, 0.08, 10),
    # },
    # "latent_bw_PAML": {
    #     "planning_depth": [1, 4, 8],
    #     "replay_capacity": [0],
    #     "lr": [0.01],
    #     "lr_p": [0.01],
    #     "lr_m": np.linspace(0.001, 0.01, 10),
    # },

}
