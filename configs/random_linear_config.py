import numpy as np
env_config = {
    "class": "Random",
    "non_gridworld": True,
    "model_class": "linear",
    "env_type": "discrete",
    "policy_type": "greedy",
    "obs_type": "dependent_features",
    "mdp_filename": None,
    "env_size": 10,
    "obs_size": 10,
    "num_episodes": 100,
    "num_runs": 5,
    "stochastic": False,
    "feature_coder": None,
    "nA": 2
}

volatile_agent_config = {
    "vanilla": {
        "planning_depth": [0],
        "replay_capacity": [0],
        "lr": np.linspace(0.1, 1.0, 10),
        "lr_p": [0],
        "lr_m": [0]
    },
    "bw_intr": {
        "planning_depth": [1],
        "replay_capacity": [0],
        "lr": [1.0],
        "lr_p": [1.0],
        "lr_m": np.linspace(0.001, 0.1, 10),
    },
    "bw_vaware": {
        "planning_depth": [1],
        "replay_capacity": [0],
        "lr": [1.0],
        "lr_p": [1.0],
        "lr_m": np.linspace(0.001, 0.1, 10),
    },
    "bw_meta": {
        "planning_depth": [1],
        "replay_capacity": [0],
        "lr": [1.0],
        "lr_p": [1.0],
        "lr_m": np.linspace(0.001, 0.1, 10),
    },
    "fw": {
        "planning_depth": [1],
        "replay_capacity": [0],
        "lr": [0.114],
        "lr_p": [0.114],
        "lr_m": np.linspace(0.01, 1.0, 10),
    },
    "bw": {
        "planning_depth": [1, 2, 3, 4],
        "replay_capacity": [0],
        "lr": [0.114],
        "lr_p": [0.114],
        "lr_m": np.linspace(0.01, 1.0, 10),
    },
    "bw_fw": {
        "planning_depth": [1],
        "replay_capacity": [0],
        "lr": [0.114],
        "lr_p": [0.114],
        "lr_m": np.linspace(0.01, 1.0, 10),
    },
    "bw_fw_PWMA": {
        "planning_depth": [1],
        "replay_capacity": [50],
        "lr": [0.114],
        "lr_p": [0.114],
        "lr_m": np.linspace(0.01, 1.0, 10),
    },
    "bw_fw_MG": {
        "planning_depth": [1],
        "replay_capacity": [50],
        "lr": [0.114],
        "lr_p": [0.114],
        "lr_m": np.linspace(0.01, 1.0, 10),
    },
    "fw_rnd": {
        "planning_depth": [1],
        "replay_capacity": [50],
        "lr": [0.114],
        "lr_p": [0.114],
        "lr_m": np.linspace(0.01, 1.0, 10),
    },
    "fw_pri": {
        "planning_depth": [1],
        "replay_capacity": [50],
        "lr": [0.114],
        "lr_p": [0.114],
        "lr_m": np.linspace(0.01, 1.0, 10),
    },
}