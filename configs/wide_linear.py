import numpy as np
env_config = {
    "class": "BipartiteLinear",
    "non_gridworld": True,
    "model_class": "linear",
    "env_type": "discrete",
    "obs_type": "position",
    "mdp_filename": None,
    "policy_type": "multinomial",
    "env_size": (500, 100, 50, 10, 5),
    "num_episodes": 10000,
    "num_runs": 1,
    "stochastic": False,
    "feature_coder": None,
    "nA": 1
}

volatile_agent_config = {
    "vanilla": {
        "planning_depth": [0],
        "replay_capacity": [0],
        "lr": np.array([0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]),
        "lr_p": [0],
        "lr_m": [0],
        "lr_ctrl": 0.4
    },
    "c_bw_PAML": {
        "planning_depth": [1],
        "replay_capacity": [0],
        "lr": [1.0],
        "lr_p": [1.0],
        "lr_m": np.array([0.001, 0.005, 0.01]),
        "lr_ctrl": 0.4
    },
    "p_fw_PAML": {
        "planning_depth": [1],
        "replay_capacity": [0],
        "lr": [0.114],
        "lr_p": [0.114],
        "lr_m": np.array([0.001, 0.005, 0.01]),
        "lr_ctrl": 0.4
    },
    "p_fw": {
        "planning_depth": [1],
        "replay_capacity": [0],
        "lr": [0.114],
        "lr_p": [0.114],
        "lr_m": np.array([0.0001, 0.0005, 0.001, 0.005]),
        "lr_ctrl": 0.4
    },
    "c_bw": {
        "planning_depth": [1],
        "replay_capacity": [0],
        "lr": [0.01],
        "lr_p": [0.01],
        "lr_m": np.array([0.001, 0.005, 0.01]),
        "lr_ctrl": 0.4
    },
    "c_true_bw": {
        "planning_depth": [1],
        "replay_capacity": [0],
        "lr": [0.01],
        "lr_p": [0.01],
        "lr_m": np.array([0.001, 0.005, 0.01]),
        "lr_ctrl": 0.4
    },
    # "bw_update": {
    #     "planning_depth": [1],
    #     "replay_capacity": [0],
    #     "lr": [1.0],
    #     "lr_p": [1.0],
    #     "lr_m": np.array([0.0001, 0.0005, 0.001, 0.005, 0.01]),
    #     "lr_ctrl": 0.4
    # },
    # "bw_mult_PAML": {
    #     "planning_depth": [1],
    #     "replay_capacity": [0],
    #     "lr": [1.0],
    #     "lr_p": [1.0],
    #     "lr_m": np.array([0.0001, 0.0005, 0.001, 0.005, 0.01]),
    #     "lr_ctrl": 0.4
    # },

    # "mult_fw": {
    #     "planning_depth": [1],
    #     "replay_capacity": [0],
    #     "lr": [0.114],
    #     "lr_p": [0.114],
    #     "lr_m": np.array([0.0001, 0.0005, 0.001, 0.005, 0.01]),
    #     "lr_ctrl": 0.4
    # },

    # "fw_mult_PAML": {
    #     "planning_depth": [1],
    #     "replay_capacity": [0],
    #     "lr": [0.114],
    #     "lr_p": [0.114],
    #     "lr_m": np.array([0.0001, 0.0005, 0.001, 0.005, 0.01]),
    #     "lr_ctrl": 0.4
    # },

    # "mult_bw": {
    #     "planning_depth": [1],
    #     "replay_capacity": [0],
    #     "lr": [0.01],
    #     "lr_p": [0.01],
    #     "lr_m": np.array([0.0001, 0.0005, 0.001, 0.005, 0.01]),
    #     "lr_ctrl": 0.4
    # },

    # "true_mult_bw": {
    #     "planning_depth": [1],
    #     "replay_capacity": [0],
    #     "lr": [0.114],
    #     "lr_p": [0.114],
    #      "lr_m": np.array([0.0001, 0.0005, 0.001, 0.005, 0.01]),
    #     "lr_ctrl": 0.4
    # },
    # "ac_vanilla": {
    #     "planning_depth": [0],
    #     "replay_capacity": [0],
    #     "lr": np.linspace(0.01, 0.1, 10),
    #     "lr_p": [0],
    #     "lr_m": [0]
    # },
    # "vanilla_PAML": {
    #     "planning_depth": [0],
    #     "replay_capacity": [0],
    #     "lr": np.linspace(0.01, 0.1, 10),
    #     "lr_p": [0],
    #     "lr_m": [0]
    # },
    # "latent_vanilla_PAML": {
    #     "planning_depth": [0],
    #     "replay_capacity": [0],
    #     "lr": np.linspace(0.01, 0.1, 10),
    #     "lr_p": [0],
    #     "lr_m": [0]
    # },
    # "bw_PAML": {
    #     "planning_depth": [1, 4, 8],
    #     "replay_capacity": [0],
    #     "lr": [0.08],
    #     "lr_p": [0.08],
    #     "lr_m": np.linspace(0.001, 0.08, 10),
    # },
    # "latent_bw_PAML": {
    #     "planning_depth": [1, 4, 8],
    #     "replay_capacity": [0],
    #     "lr": [0.01],
    #     "lr_p": [0.01],
    #     "lr_m": np.linspace(0.001, 0.01, 10),
    # },

}