import numpy as np
env_config = {
    "class": "Boyan",
    "non_gridworld": True,
    "model_class": "linear",
    "env_type": "discrete",
    "obs_type": "spikes",
    "policy_type": "greedy",
    "env_size": 98,
    "obs_size": 25,
    "num_episodes": 100,
    "num_runs": 1,
    "stochastic": False,
    "nA": 1
}

volatile_agent_config = {
    "vanilla": {
        "planning_depth": [0],
        "replay_capacity": [0],
        "lr": np.linspace(0.01, 1.0, 10),
        "lr_p": [0],
        "lr_m": [0]
    },
    "bw_intr": {
        "planning_depth": [1, 2, 3, 4],
        "replay_capacity": [0],
        "lr": [1.0],
        "lr_p": [1.0],
        "lr_m": np.linspace(0.01, 1.0, 10),
    },
    "latent_bw_intr": {
        "planning_depth": [1, 2, 3, 4],
        "replay_capacity": [0],
        "lr": [1.0],
        "lr_p": [1.0],
        "lr_m": np.linspace(0.01, 1.0, 10),
    },
    "bw": {
        "planning_depth": [1, 2, 3, 4],
        "replay_capacity": [0],
        "lr": [1.0],
        "lr_p": [1.0],
        "lr_m": np.linspace(0.01, 1.0, 10),
    },
    "fw": {
        "planning_depth": [1, 2, 3, 4],
        "replay_capacity": [0],
        "lr": [1.0],
        "lr_p": [1.0],
        "lr_m": np.linspace(0.01, 1.0, 10),
    },
}