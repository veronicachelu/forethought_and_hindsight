import numpy as np
env_config = {
    "class": "MicroWorld",
    "non_gridworld": False,
    "model_class": "linear",
    "env_type": "discrete",
    "obs_type": "onehot",
    "policy_type": "greedy",
    "mdp_filename": "./mdps/maze_80.mdp",
    "env_size": 80,
    "num_episodes": 100,
    "num_runs": 100,
    "stochastic": False,
    "feature_coder": None,
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
        "planning_depth": [1, 4, 8],
        "replay_capacity": [0],
        "lr": [0.01],
        "lr_p": [0.01],
        "lr_m": np.linspace(0.001, 0.1, 10),
    },
    "bw_vaware": {
        "planning_depth": [1],
        "replay_capacity": [0],
        "lr": [0.01],
        "lr_p": [0.01],
        "lr_m": np.linspace(0.0001, 0.01, 10),
    },
    "bw_meta": {
        "planning_depth": [1],
        "replay_capacity": [0],
        "lr": [0.01],
        "lr_p": [0.01],
        "lr_m": np.linspace(0.0001, 0.01, 10),
    },
     "bw_intr": {
        "planning_depth": [1],
        "replay_capacity": [0],
        "lr": [0.08],
        "lr_p": [0.08],
        "lr_m": np.linspace(0.001, 0.1, 10),
    },
    "fw": {
        "planning_depth": [1],
        "replay_capacity": [0],
        "lr": [0.01],
        "lr_p": [0.01],
        "lr_m": np.linspace(0.01, 1.0, 10),
    },
    "fw_rnd": {
        "planning_depth": [1],
        "replay_capacity": [50],
        "lr": [0.01],
        "lr_p": [0.01],
        "lr_m": np.linspace(0.01, 1.0, 10),
    },
    # "ac_vanilla": {
    #     "planning_depth": [0],
    #     "replay_capacity": [0],
    #     "lr": np.linspace(0.01, 0.1, 10),
    #     "lr_p": [0],
    #     "lr_m": [0]
    # },
    # "vanilla_intr": {
    #     "planning_depth": [0],
    #     "replay_capacity": [0],
    #     "lr": np.linspace(0.01, 0.1, 10),
    #     "lr_p": [0],
    #     "lr_m": [0]
    # },
    # "latent_vanilla_intr": {
    #     "planning_depth": [0],
    #     "replay_capacity": [0],
    #     "lr": np.linspace(0.01, 0.1, 10),
    #     "lr_p": [0],
    #     "lr_m": [0]
    # },
    # "bw_intr": {
    #     "planning_depth": [1, 4, 8],
    #     "replay_capacity": [0],
    #     "lr": [0.08],
    #     "lr_p": [0.08],
    #     "lr_m": np.linspace(0.001, 0.08, 10),
    # },
    # "latent_bw_intr": {
    #     "planning_depth": [1, 4, 8],
    #     "replay_capacity": [0],
    #     "lr": [0.01],
    #     "lr_p": [0.01],
    #     "lr_m": np.linspace(0.001, 0.01, 10),
    # },

}