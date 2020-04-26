import numpy as np
env_config = {
    "class": "DMEnvFromGym",
    "non_gridworld": True,
    "model_class": "linear",
    "feature_coder": {
        "type": "rbf"
    },
    "env_type": "continuous",
    "obs_type": "position",
    "policy_type": "estimated",
    "mdp_filename": "CartPole-v0",
    "num_episodes": 50,
    "control_num_episodes": 100,
    "num_runs": 20,
    "stochastic": False,
    "nA": 2,
    "env_size": 4,

}

volatile_agent_config = {
    "vanilla": {
        "planning_depth": [0],
        "replay_capacity": [0],
        "lr": np.linspace(0.1, 0.8, 10),
        "lr_p": [0],
        "lr_m": [0],
        "lr_ctrl": 0.4
    },
    "bw": {
        "planning_depth": [1],
        "replay_capacity": [0],
        "lr": [0.01],
        "lr_p": [0.01],
        "lr_m": np.linspace(0.005, 0.05, 10),
        "lr_ctrl": 0.4
    },
    "fw": {
        "planning_depth": [1],
        "replay_capacity": [0],
        "lr": [0.01],
        "lr_p": [0.01],
        "lr_m": np.linspace(0.005, 0.05, 10),
        "lr_ctrl": 0.4
    },
    "fw_intr": {
        "planning_depth": [1],
        "replay_capacity": [0],
        "lr": [0.01],
        "lr_p": [0.01],
        "lr_m": np.linspace(0.005, 0.05, 10),
        "lr_ctrl": 0.4
    },
    "fw_rnd": {
        "planning_depth": [1],
        "replay_capacity": [50],
        "lr": [0.01],
        "lr_p": [0.01],
        "lr_m": np.linspace(0.001, 0.07, 10),
        "lr_ctrl": 0.4
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
    "bw_intr": {
        "planning_depth": [1],
        "replay_capacity": [0],
        "lr": [0.08],
        "lr_p": [0.08],
        "lr_m": np.linspace(0.005, 0.05, 10),
    },
    # "latent_bw_intr": {
    #     "planning_depth": [1, 4, 8],
    #     "replay_capacity": [0],
    #     "lr": [0.01],
    #     "lr_p": [0.01],
    #     "lr_m": np.linspace(0.001, 0.01, 10),
    # },

}