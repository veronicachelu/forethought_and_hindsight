import numpy as np

env_config = {
    "class": "MicroWorld",
    "non_gridworld": False,
    "model_class": "tabular",
    "model_family": "extrinsic",
    "env_type": "discreate",
    "obs_type": "tabular",
    "policy_type": "greedy",
    "mdp_filename": "./mdps/maze_486.mdp",
    "env_size": 486,
    "num_episodes": 100,
    "num_runs": 100,
    "stochastic": True,
    "nA": 4
}

volatile_agent_config = {
    "vanilla": {
        "planning_depth": [0],
        "replay_capacity": [0],
        "lr": np.linspace(0.1, 1.0, 10),
        "lr_p": [0],
        "lr_m": [0]
    },
    "bw": {
        "planning_depth": [1, 2, 3, 4],
        "replay_capacity": [0],
        "lr": [0.7],
        "lr_p": [0.7],
        "lr_m": np.linspace(0.1, 1.0, 10),
    },
    "bw_fw": {
        "planning_depth": [1],
        "replay_capacity": [0],
        "lr": [1.0],
        "lr_p": np.linspace(0.1, 1.0, 10),
        "lr_m": np.linspace(0.1, 1.0, 10),
    },
    "bw_fw_PWMA": {
        "planning_depth": [1],
        "replay_capacity": [50],
        "lr": [1.0],
        "lr_p": np.linspace(0.1, 1.0, 10),
        "lr_m": np.linspace(0.1, 1.0, 10),
    },
    "bw_fw_MG": {
        "planning_depth": [1],
        "replay_capacity": [50],
        "lr": [1.0],
        "lr_p": np.linspace(0.1, 1.0, 10),
        "lr_m": np.linspace(0.1, 1.0, 10),
    },
    "fw_rnd": {
        "planning_depth": [1],
        "replay_capacity": [50],
        "lr": [1.0],
        "lr_p": np.linspace(0.1, 1.0, 10),
        "lr_m": np.linspace(0.1, 1.0, 10),
    },
    "fw_pri": {
        "planning_depth": [1],
        "replay_capacity": [50],
        "lr": [1.0],
        "lr_p": np.linspace(0.1, 1.0, 10),
        "lr_m": np.linspace(0.1, 1.0, 10),
    },
}