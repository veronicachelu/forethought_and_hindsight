import numpy as np
env_config = {
    "class": "RandomChain",
    "non_gridworld": True,
    "model_class": "tabular",
    "env_type": "discreate",
    "obs_type": "tabular",
    "mdp_filename": None,
    "env_size": 10,
    "num_episodes": 100,
    "num_runs": 100,
    "stochastic": False,
    "nA": 2
}

volatile_agent_config = {
    "vanilla": {
        "planning_depth": [0],
        "replay_capacity": [0],
        "lr": np.linspace(0.1, 1.0, 10),
        "lr_m": [0]
    },
    "bw": {
        "planning_depth": [1, 2, 3, 4],
        "replay_capacity": [0],
        "lr": np.linspace(0.1, 1.0, 10),
        "lr_m": np.linspace(0.1, 1.0, 10),
    },
    "bw_fw": {
        "planning_depth": [1],
        "replay_capacity": [0],
        "lr": np.linspace(0.1, 1.0, 10),
        "lr_m": np.linspace(0.1, 1.0, 10),
    },
    "bw_fw_PWMA": {
        "planning_depth": [1],
        "replay_capacity": [50],
        "lr": np.linspace(0.1, 1.0, 10),
        "lr_m": np.linspace(0.1, 1.0, 10),
    },
    "bw_fw_MG": {
        "planning_depth": [1],
        "replay_capacity": [50],
        "lr": np.linspace(0.1, 1.0, 10),
        "lr_m": np.linspace(0.1, 1.0, 10),
    },
    "fw_rnd": {
        "planning_depth": [0],
        "replay_capacity": [50],
        "lr": np.linspace(0.1, 1.0, 10),
        "lr_m": [0]
    },
    "fw_pri": {
        "planning_depth": [0],
        "replay_capacity": [50],
        "lr": np.linspace(0.1, 1.0, 10),
        "lr_m": [0]
    },
}