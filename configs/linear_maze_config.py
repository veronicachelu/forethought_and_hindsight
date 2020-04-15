import numpy as np
env_config = {
    "class": "MicroWorld",
    "non_gridworld": False,
    "model_class": "linear",
    "model_family": "extrinsic",
    "env_type": "discreate",
    "obs_type": "onehot",
    "policy_type": "greedy",
    "mdp_filename": "./mdps/maze_80.mdp",
    "env_size": 80,
    "num_episodes": 100,
    "num_runs": 100,
    "stochastic": False,
    "nA": 4
}

volatile_agent_config = {
    "vanilla_intr": {
        "planning_depth": [0],
        "replay_capacity": [0],
        "lr": np.linspace(0.01, 0.1, 10),
        "lr_p": [0],
        "lr_m": [0]
    },
    "latent_vanilla_intr": {
        "planning_depth": [0],
        "replay_capacity": [0],
        "lr": np.linspace(0.01, 0.1, 10),
        "lr_p": [0],
        "lr_m": [0]
    },
    "vanilla": {
        "planning_depth": [0],
        "replay_capacity": [0],
        "lr": np.linspace(0.01, 0.1, 10),
        "lr_p": [0],
        "lr_m": [0]
    },
    "bw_intr": {
        "planning_depth": [1, 2, 3, 4],
        "replay_capacity": [0],
        "lr": [0.03],
        "lr_p": [0.03],
        "lr_m": np.linspace(0.1, 1.0, 10),
    },
    "latent_bw_intr": {
        "planning_depth": [1, 2, 3, 4],
        "replay_capacity": [0],
        "lr": [0.01],
        "lr_p": [0.01],
        "lr_m": np.linspace(0.1, 1.0, 10),
    },
    "bw": {
        "planning_depth": [1, 2, 3, 4],
        "replay_capacity": [0],
        "lr": [0.1],
        "lr_p": [0.1],
        "lr_m": np.linspace(0.1, 1.0, 10),
    },
    "bw_fw": {
        "planning_depth": [1],
        "replay_capacity": [0],
        "lr": [0.1],
        "lr_p": [0.1],
        "lr_m": np.linspace(0.1, 1.0, 10),
    },
    "bw_fw_PWMA": {
        "planning_depth": [1],
        "replay_capacity": [50],
        "lr": [0.1],
        "lr_p": [0.1],
        "lr_m": np.linspace(0.1, 1.0, 10),
    },
    "bw_fw_MG": {
        "planning_depth": [1],
        "replay_capacity": [50],
        "lr": [0.1],
        "lr_p": [0.1],
        "lr_m": np.linspace(0.1, 1.0, 10),
    },
    "fw_rnd": {
        "planning_depth": [1],
        "replay_capacity": [50],
        "lr": [0.1],
        "lr_p": [0.1],
        "lr_m": np.linspace(0.1, 1.0, 10),
    },
    "fw_pri": {
        "planning_depth": [1],
        "replay_capacity": [50],
        "lr": [0.1],
        "lr_p": [0.1],
        "lr_m": np.linspace(0.1, 1.0, 10),
    },
}