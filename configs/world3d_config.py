import numpy as np
env_config = {
    "class": "World3d",
    "non_gridworld": True,
    "model_class": "tabular",
    "env_type": "discrete",
    "obs_type": "tabular",
    "mdp_filename": None,
    "policy_type": "greedy",
    "env_size": 1000,
    "num_episodes": 1,
    "num_runs": 10,
    "stochastic": False,
    "feature_coder": None,
    "max_len":  100000,
    "log_every_steps": 100,
    "nA": 6
}

volatile_agent_config = {
    "vanilla": {
        "planning_depth": [0],
        "replay_capacity": [0],
        "lr": np.linspace(0.1, 1.0, 10),
        "lr_m": [0],
        "lr_p": [0],
        "lr_ctrl": 0.4
    },
    "bw": {
        "planning_depth": [1],
        "replay_capacity": [0],
        "lr": [0.1],
        "lr_p": [0.1],
        # "lr_m": np.linspace(0.1, 0.5, 5),
        "lr_m": np.array([0.1, 0.5]),
        "lr_ctrl": 0.4
    },
    "fw": {
        "planning_depth": [1],
        "replay_capacity": [0],
        "lr": [0.1],
        "lr_p": [0.1],
        "lr_m": np.linspace(0.1, 0.5, 5),
        "lr_ctrl": 0.4
    },
    "bw_fw": {
        "planning_depth": [1],
        "replay_capacity": [0],
        "lr": [0.1],
        "lr_p": [0.1],
        "lr_m": np.linspace(0.01, 0.1, 10),
        "lr_ctrl": 0.4
    },
    "bw_fw_PWMA": {
        "planning_depth": [1],
        "replay_capacity": [50],
        "lr": [0.1],
        "lr_p": [0.1],
        "lr_m": np.linspace(0.01, 0.1, 10),
        "lr_ctrl": 0.4
    },
    "bw_fw_MG": {
        "planning_depth": [1],
        "replay_capacity": [50],
        "lr": [0.1],
        "lr_p": [0.1],
        "lr_m": np.linspace(0.01, 0.1, 10),
    },
    "fw_rnd": {
        "planning_depth": [1],
        "replay_capacity": [50],
        "lr": [0.1],
        "lr_p": [0.1],
        "lr_m": np.linspace(0.01, 0.1, 10),
        "lr_ctrl": 0.4
    },
    "fw_pri": {
        "planning_depth": [1],
        "replay_capacity": [50],
        "lr": [0.1],
        "lr_p": [0.1],
        "lr_m": np.linspace(0.01, 0.1, 10),
        "lr_ctrl": 0.4
    },
}