import numpy as np
env_config = {
    "class": "MicroWorld",
    "non_gridworld": False,
    "model_class": "tabular",
    "env_type": "discrete",
    "obs_type": "tabular",
    "policy_type": "greedy",
    "mdp_filename": "./mdps/maze_48.mdp",
    "env_size": 48,
    "num_episodes": 100,
    "control_num_episodes": 100,
    "num_runs": 10,
    "stochastic": False,
    "feature_coder": None,
    "reward_prob": 1.0,
    "max_reward": 1.0,
    "dynamics_prob": 0.8,
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
        "lr": [0.5],
        "lr_p": [0.5],
        "lr_m": np.linspace(0.01, 1.0, 9),
    },
    "fw": {
        "planning_depth": [1],
        "replay_capacity": [0],
        "lr": [0.5],
        "lr_p": [0.5],
        "lr_m": np.linspace(0.01, 1.0, 9),
    },
    "bw_fw": {
        "planning_depth": [1],
        "replay_capacity": [0],
        "lr": [0.5],
        "lr_p": [0.5],
        "lr_m": np.linspace(0.01, 1.0, 9),
    },
    "bw_fw_PWMA": {
        "planning_depth": [1],
        "replay_capacity": [50],
        "lr": [0.5],
        "lr_p": [0.5],
        "lr_m": np.linspace(0.01, 0.89, 9),
    },
    "bw_fw_MG": {
        "planning_depth": [1],
        "replay_capacity": [50],
        "lr": [0.5],
        "lr_p": [0.5],
        "lr_m": np.linspace(0.01, 0.89, 9),
    },
    "fw_rnd": {
        "planning_depth": [1],
        "replay_capacity": [50],
        "lr": [0.5],
        "lr_p": [0.5],
        "lr_m": np.linspace(0.01, 0.89, 9),
    },
    "fw_pri": {
        "planning_depth": [1],
        "replay_capacity": [50],
        "lr": [0.5],
        "lr_p": [0.5],
        "lr_m": np.linspace(0.01, 0.89, 9),
    },
}