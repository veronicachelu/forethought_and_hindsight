import numpy as np
env_config = {
    "class": "Random",
    "non_gridworld": True,
    "model_class": "linear",
    "env_type": "discrete",
    "policy_type": "random",
    "obs_type": "dependent_features",
    "mdp_filename": None,
    "env_size": 5,
    "obs_size": 5,
    "num_episodes": 200,
    "num_runs": 50,
    "stochastic": False,
    "feature_coder": None,
    "nA": 2
}

volatile_agent_config = {
    "vanilla": {
        "planning_depth": [0],
        "replay_capacity": [0],
        "lr": np.array([0.001, 0.005, 0.01, 0.05, 0.1, 0.5]),
        "lr_p": [0],
        "lr_m": [0],
        "lr_ctrl": 0.4
    },
    "bw_PAML": {
        "planning_depth": [1],
        "replay_capacity": [0],
        "lr": [1.0],
        "lr_p": [1.0],
        "lr_m": np.array([0.001, 0.005, 0.01, 0.05, 0.1, 0.5]),
        "lr_ctrl": 0.4
    },
    "bw_meta": {
        "planning_depth": [1],
        "replay_capacity": [0],
        "lr": [1.0],
        "lr_p": [1.0],
        "lr_m": np.array([0.001, 0.005, 0.01, 0.05, 0.1, 0.5]),
        "lr_ctrl": 0.4
    },
    "bw_update": {
        "planning_depth": [1],
        "replay_capacity": [0],
        "lr": [1.0],
        "lr_p": [1.0],
        "lr_m": np.array([0.001, 0.005, 0.01, 0.05, 0.1, 0.5]),
        "lr_ctrl": 0.4
    },
    "bw_mult_PAML": {
        "planning_depth": [1],
        "replay_capacity": [0],
        "lr": [1.0],
        "lr_p": [1.0],
        "lr_m": np.array([0.001, 0.005, 0.01, 0.05, 0.1, 0.5]),
        "lr_ctrl": 0.4
    },
    "fw": {
        "planning_depth": [1],
        "replay_capacity": [0],
        "lr": [0.114],
        "lr_p": [0.114],
        "lr_m": np.array([0.001, 0.005, 0.01, 0.05, 0.1, 0.5]),
        "lr_ctrl": 0.4
    },
    "mult_fw": {
        "planning_depth": [1],
        "replay_capacity": [0],
        "lr": [0.114],
        "lr_p": [0.114],
        "lr_m": np.array([0.001, 0.005, 0.01, 0.05, 0.1, 0.5]),
        "lr_ctrl": 0.4
    },
    "fw_PAML": {
        "planning_depth": [1],
        "replay_capacity": [0],
        "lr": [0.114],
        "lr_p": [0.114],
        "lr_m": np.array([0.001, 0.005, 0.01, 0.05, 0.1, 0.5]),
        "lr_ctrl": 0.4
    },
    "fw_mult_PAML": {
        "planning_depth": [1],
        "replay_capacity": [0],
        "lr": [0.114],
        "lr_p": [0.114],
        "lr_m": np.array([0.001, 0.005, 0.01, 0.05, 0.1, 0.5]),
        "lr_ctrl": 0.4
    },
    "bw": {
        "planning_depth": [1],
        "replay_capacity": [0],
        "lr": [0.114],
        "lr_p": [0.114],
        "lr_m": np.array([0.001, 0.005, 0.01, 0.05, 0.1, 0.5]),
        "lr_ctrl": 0.4
    },
    "mult_bw": {
        "planning_depth": [1],
        "replay_capacity": [0],
        "lr": [0.114],
        "lr_p": [0.114],
        "lr_m": np.array([0.001, 0.005, 0.01, 0.05, 0.1, 0.5]),
        "lr_ctrl": 0.4
    },
    "true_bw": {
        "planning_depth": [1],
        "replay_capacity": [0],
        "lr": [0.114],
        "lr_p": [0.114],
        "lr_m": np.array([0.001, 0.005, 0.01, 0.05, 0.1, 0.5]),
        "lr_ctrl": 0.4
    },
    "true_mult_bw": {
        "planning_depth": [1],
        "replay_capacity": [0],
        "lr": [0.114],
        "lr_p": [0.114],
        "lr_m": np.array([0.001, 0.005, 0.01, 0.05, 0.1, 0.5]),
        "lr_ctrl": 0.4
    },
}