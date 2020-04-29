configs={
    "maze": {
        "non_parametric_fw": {
            "agents": ["fw_rnd", "fw_pri"],
            "planning_depths": [1, 1],
            "replay_capacities": [50, 50]
        },

        "non_parametric_fw_bw": {
            "agents": ["bw_fw_PWMA", "bw_fw_MG"],
            "planning_depths": [1, 1],
            "replay_capacities": [50, 50]
        },

        "parametric": {
            "agents": ["bw", "bw_fw", "fw"],
            "planning_depths": [1, 1, 1],
            "replay_capacities": [0, 0, 0],
        },

        "final": {
            "agents": ["bw", "fw", "fw_rnd", "bw_fw_PWMA"],
            "planning_depths": [1, 1, 1, 1],
            "replay_capacities": [0, 0, 50, 50]
        },
        "all": {
            "agents": ["bw", "bw_fw", "fw", "fw_rnd", "fw_pri",
                       "bw_fw_PWMA", "bw_fw_MG"],
            "planning_depths": [1, 1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 50, 50, 50, 50]
        }
    },
    "stoch_linear_maze": {
        "all": {
            "agents": ["bw", "bw_intr", "fw", "fw_intr"],
            "planning_depths": [1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0]
        }
    },
    "linear_maze": {
        "bw": {
            "agents": ["bw_intr", "vanilla_intr"],
            "planning_depths": [1, 0],
            "replay_capacities": [0, 0]
        },
        "true_bw": {
            "agents": ["true_bw", "bw"],
            "planning_depths": [1, 1],
            "replay_capacities": [0, 0]
        },
        "fw": {
            "agents": ["fw_intr", "vanilla_intr"],
            "planning_depths": [1, 0],
            "replay_capacities": [0, 0]
        },
        "bw_fw": {
            "agents": ["bw_intr", "fw_intr", "vanilla_intr"],
            "planning_depths": [1, 1, 0],
            "replay_capacities": [0, 0, 0]
        },
        "corr_vs_value_vs_meta": {
            "agents": ["bw_intr", "bw_vaware", "bw_meta"],
            "planning_depths": [1, 1, 1],
            "replay_capacities": [0, 0, 0]
        },
        "intr_vs_extr": {
            "agents": ["bw_intr", "bw", "vanilla_intr"],
            "planning_depths": [1, 1, 0],
            "replay_capacities": [0, 0, 0]
        },
        "intr_vs_extr_fw_and_bw": {
            "agents": ["bw_intr", "bw", "fw_intr", "fw", "vanilla_intr"],
            "planning_depths": [1, 1, 1, 1, 0],
            "replay_capacities": [0, 0, 0, 0, 0]
        },
        "latent_vs_no_latent": {
            "agents": ["bw_intr", "latent_bw_intr", "vanilla_intr"],
            "planning_depths": [1, 1, 0],
            "replay_capacities": [0, 0, 0]
        },
        "random_vs_learned": {
            "agents": ["random", "learned"],
            "planning_depths": [1, 1, 0],
            "replay_capacities": [0, 0, 0]
        },
        "final": {
            "agents": ["bw", "fw", "fw_rnd"],
            "planning_depths": [1, 1, 1],
            "replay_capacities": [0, 0, 50]
        },
        "all": {
            "agents": ["bw", "bw_intr", "fw", "fw_intr"],
            "planning_depths": [1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0]
        }
    },
    "random_linear": {
        "latent_vs_no_latent": {
            "agents": ["bw_intr", "latent_bw_intr", "vanilla_intr"],
            "planning_depths": [1, 1, 0],
            "replay_capacities": [0, 0, 0]
        },
        "bw_fw": {
            "agents": ["bw_intr", "fw_intr", "vanilla_intr"],
            "planning_depths": [1, 1, 0],
            "replay_capacities": [0, 0, 0]
        },
        "all": {
            "agents": ["bw", "mult_bw",
                       "true_bw", "true_mult_bw",
                       "fw", "fw_intr",
                       "bw_intr", "bw_mult_intr"],
            "planning_depths": [1, 1, 1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0, 0, 0, 0, 0]
        },
        "corr_vs_value_vs_meta": {
            "agents": ["bw_intr", "bw_vaware", "bw_meta"],
            "planning_depths": [1, 1, 1],
            "replay_capacities": [0, 0, 0]
        },
    },
    "boyan": {
        "latent_vs_no_latent": {
            "agents": ["bw_intr", "latent_bw_intr", "vanilla_intr"],
            "planning_depths": [1, 1, 0],
            "replay_capacities": [0, 0, 0]
        },
        "bw_fw": {
            "agents": ["bw_intr", "fw_intr", "vanilla_intr"],
            "planning_depths": [1, 1, 0],
            "replay_capacities": [0, 0, 0]
        },
        "intr_vs_extr_fw_and_bw": {
            "agents": ["bw_intr", "bw", "fw_intr", "fw", "vanilla_intr"],
            "planning_depths": [1, 1, 1, 1, 0],
            "replay_capacities": [0, 0, 0, 0, 0]
        },
        "intr_vs_extr": {
            "agents": ["bw_intr", "bw"],
            "planning_depths": [1, 1],
            "replay_capacities": [0, 0]
        },
        "all": {
            "agents": ["bw", "fw", "fw_rnd"],
            "planning_depths": [1, 1, 1],
            "replay_capacities": [0, 0, 50]
        },
        "corr_vs_value_vs_meta": {
            "agents": ["bw_intr", "bw_vaware", "bw_meta"],
            "planning_depths": [1, 1, 1],
            "replay_capacities": [0, 0, 0]
        },
    },
    "medium_maze": {
        "final": {
            "agents": ["bw", "fw", "fw_rnd", "bw_fw_PWMA"],
            "planning_depths": [1, 1, 1, 1],
            "replay_capacities": [0, 0, 50, 50]
        },
        "all": {
            "agents": ["bw", "fw", "fw_rnd", "bw_fw_PWMA"],
            "planning_depths": [1, 1, 1, 1],
            "replay_capacities": [0, 0, 50, 50]
        }
    },
    "random": {
        "non_parametric_fw": {
            "agents": ["fw_rnd", "fw_pri"],
            "planning_depths": [1, 1],
            "replay_capacities": [50, 50]
        },

        "non_parametric_fw_bw": {
            "agents": ["bw_fw_PWMA", "bw_fw_MG"],
            "planning_depths": [1, 1],
            "replay_capacities": [50, 50]
        },

        "parametric": {
            "agents": ["bw", "bw_fw", "fw"],
            "planning_depths": [1, 1, 1],
            "replay_capacities": [0, 0, 0],
        },

        "final": {
            "agents": ["bw", "fw", "fw_rnd", "bw_fw_PWMA"],
            "planning_depths": [1, 1, 1, 1],
            "replay_capacities": [0, 0, 50, 50]
        },
        "all": {
            "agents": ["bw", "bw_fw", "fw", "fw_rnd", "fw_pri",
                       "bw_fw_PWMA", "bw_fw_MG"],
            "planning_depths": [1, 1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 50, 50, 50, 50]
        }
        },
    "loop": {
        "non_parametric_fw": {
            "agents": ["fw_rnd", "fw_pri"],
            "planning_depths": [1, 1],
            "replay_capacities": [50, 50]
        },

        "non_parametric_fw_bw": {
            "agents": ["bw_fw_PWMA", "bw_fw_MG"],
            "planning_depths": [1, 1],
            "replay_capacities": [50, 50]
        },

        "parametric": {
            "agents": ["bw", "bw_fw", "fw"],
            "planning_depths": [1, 1, 1],
            "replay_capacities": [0, 0, 0],
        },

        "final": {
            "agents": ["bw", "fw", "fw_rnd", "bw_fw_PWMA"],
            "planning_depths": [1, 1, 1, 1],
            "replay_capacities": [0, 0, 50, 50]
        },
        "all": {
            "agents": ["bw", "bw_fw", "fw", "fw_rnd", "fw_pri",
                       "bw_fw_PWMA", "bw_fw_MG"],
            "planning_depths": [1, 1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 50, 50, 50, 50]
        }
        },
    "repeat": {
        "non_parametric_fw": {
            "agents": ["fw_rnd", "fw_pri"],
            "planning_depths": [1, 1],
            "replay_capacities": [50, 50]
        },

        "non_parametric_fw_bw": {
            "agents": ["bw_fw_PWMA", "bw_fw_MG"],
            "planning_depths": [1, 1],
            "replay_capacities": [50, 50]
        },

        "parametric": {
            "agents": ["bw", "bw_fw", "fw"],
            "planning_depths": [1, 1, 1],
            "replay_capacities": [0, 0, 0],
        },

        "final": {
            "agents": ["bw", "fw", "fw_rnd", "bw_fw_PWMA"],
            "planning_depths": [1, 1, 1, 1],
            "replay_capacities": [0, 0, 50, 50]
        },
        "all": {
             "agents": ["bw", "bw_fw", "fw", "fw_rnd", "fw_pri",
                       "bw_fw_PWMA", "bw_fw_MG"],
            "planning_depths": [1, 1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 50, 50, 50, 50]
        }
        },
    "shortcut": {
        "non_parametric_fw": {
            "agents": ["fw_rnd", "fw_pri"],
            "planning_depths": [1, 1],
            "replay_capacities": [50, 50]
        },

        "non_parametric_fw_bw": {
            "agents": ["bw_fw_PWMA", "bw_fw_MG"],
            "planning_depths": [1, 1],
            "replay_capacities": [50, 50]
        },

        "parametric": {
            "agents": ["bw", "bw_fw", "fw"],
            "planning_depths": [1, 1, 1],
            "replay_capacities": [0, 0, 0],
        },

        "final": {
            "agents": ["bw", "fw", "fw_rnd", "bw_fw_PWMA"],
            "planning_depths": [1, 1, 1, 1],
            "replay_capacities": [0, 0, 50, 50]
        },
        "all": {
            "agents": ["bw", "bw_fw", "fw", "fw_rnd", "fw_pri",
                       "bw_fw_PWMA", "bw_fw_MG"],
            "planning_depths": [1, 1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 50, 50, 50, 50]
        }
        },
    "cartpole": {
        "all": {
            "agents": ["bw", "fw", "fw_rnd"],
            "planning_depths": [1, 1, 1],
            "replay_capacities": [0, 0, 50]
        },
        "latent_vs_no_latent": {
            "agents": ["bw_intr", "latent_bw_intr", "vanilla_intr"],
            "planning_depths": [1, 1, 0],
            "replay_capacities": [0, 0, 0]
        },
        "bw_fw": {
            "agents": ["bw_intr", "fw_intr", "vanilla_intr"],
            "planning_depths": [1, 1, 0],
            "replay_capacities": [0, 0, 0]
        },
        "intr_vs_extr_fw_and_bw": {
            "agents": ["bw_intr", "bw", "fw_intr", "fw", "vanilla_intr"],
            "planning_depths": [1, 1, 1, 1, 0],
            "replay_capacities": [0, 0, 0, 0, 0]
        },
        "intr_vs_extr": {
            "agents": ["bw_intr", "bw"],
            "planning_depths": [1, 1],
            "replay_capacities": [0, 0]
        },
        "corr_vs_value_vs_meta": {
            "agents": ["bw_intr", "bw_vaware", "bw_meta"],
            "planning_depths": [1, 1, 1],
            "replay_capacities": [0, 0, 0]
        },
    },
    "split": {
        "all": {
            "agents": ["bw", "bw_fw", "fw", "fw_rnd", "fw_pri",
                       "bw_fw_PWMA", "bw_fw_MG"],
            "planning_depths": [1, 1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 50, 50, 50, 50]
        }
    }
}