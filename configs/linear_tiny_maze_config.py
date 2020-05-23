import numpy as np
env_config = {
    "class": "MicroWorld",
    "non_gridworld": False,
    "model_class": "linear",
    "feature_coder": None,
    "env_type": "discrete",
    "obs_type": "onehot",
    "policy_type": "greedy",
    "mdp_filename": "./mdps/maze_48.mdp",
    "env_size": 48,
    "num_episodes": 100,
    "control_num_episodes": 500,
    "num_runs": 1,
    "stochastic": False,
    "nA": 4
}

volatile_agent_config = {
    "vanilla": {
        "planning_depth": [0],
        "replay_capacity": [0],
        "lr": np.array([0.005, 0.01, 0.05, 0.1, 0.5, 1.0]),
        "lr_p": [0],
        "lr_m": [0],
        "lr_ctrl": 0.4
    },
    "c_bw_PAML": {
        "planning_depth": [1],
        "replay_capacity": [0],
        "lr": [1.0],
        "lr_p": [1.0],
        "lr_m": np.array([0.005, 0.01, 0.05, 0.1]),
        "lr_ctrl": 0.4
    },
    "p_bw_PAML": {
        "planning_depth": [1],
        "replay_capacity": [0],
        "lr": [1.0],
        "lr_p": [1.0],
        "lr_m": np.array([0.005, 0.01, 0.05, 0.1]),
        "lr_ctrl": 0.4
    },
    "p_fw_PAML": {
        "planning_depth": [1],
        "replay_capacity": [0],
        "lr": [0.114],
        "lr_p": [0.114],
        "lr_m": np.array([0.005, 0.01, 0.05, 0.1]),
        "lr_ctrl": 0.4
    },
    "p_fw": {
        "planning_depth": [1],
        "replay_capacity": [0],
        "lr": [0.114],
        "lr_p": [0.114],
        "lr_m": np.array([0.005, 0.01, 0.05, 0.1]),
        "lr_ctrl": 0.4
    },
    "c_bw": {
        "planning_depth": [1],
        "replay_capacity": [0],
        "lr": [0.01],
        "lr_p": [0.01],
        "lr_m": np.array([0.005, 0.01, 0.05, 0.1]),
        "lr_ctrl": 0.4
    },
    "p_bw": {
        "planning_depth": [1],
        "replay_capacity": [0],
        "lr": [0.01],
        "lr_p": [0.01],
        "lr_m": np.array([0.005, 0.01, 0.05, 0.1]),
        "lr_ctrl": 0.4
    },
    "c_true_bw": {
        "planning_depth": [1],
        "replay_capacity": [0],
        "lr": [0.01],
        "lr_p": [0.01],
        "lr_m": np.array([0.005, 0.01, 0.05, 0.1]),
        "lr_ctrl": 0.4
    },
    "p_true_bw": {
        "planning_depth": [1],
        "replay_capacity": [0],
        "lr": [0.01],
        "lr_p": [0.01],
        "lr_m": np.array([0.005, 0.01, 0.05, 0.1]),
        "lr_ctrl": 0.4
    },
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