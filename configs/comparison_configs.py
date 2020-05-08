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
            "agents": ["bw", "bw_PAML", "fw", "fw_PAML"],
            "planning_depths": [1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0]
        }
    },
    "linear_maze": {
        "bw": {
            "agents": ["bw_PAML", "vanilla_PAML"],
            "planning_depths": [1, 0],
            "replay_capacities": [0, 0]
        },
        "true_bw": {
            "agents": ["true_bw", "bw"],
            "planning_depths": [1, 1],
            "replay_capacities": [0, 0]
        },
        "fw": {
            "agents": ["fw_PAML", "vanilla_PAML"],
            "planning_depths": [1, 0],
            "replay_capacities": [0, 0]
        },
        "bw_fw": {
            "agents": ["bw_PAML", "fw_PAML", "vanilla_PAML"],
            "planning_depths": [1, 1, 0],
            "replay_capacities": [0, 0, 0]
        },
        "corr_vs_value_vs_meta": {
            "agents": ["bw_PAML", "bw_vaware", "bw_meta"],
            "planning_depths": [1, 1, 1],
            "replay_capacities": [0, 0, 0]
        },
        "PAML_vs_MLE": {
            "agents": ["bw_PAML", "bw", "vanilla_PAML"],
            "planning_depths": [1, 1, 0],
            "replay_capacities": [0, 0, 0]
        },
        "PAML_vs_MLE_fw_and_bw": {
            "agents": ["bw_PAML", "bw", "fw_PAML", "fw", "vanilla_PAML"],
            "planning_depths": [1, 1, 1, 1, 0],
            "replay_capacities": [0, 0, 0, 0, 0]
        },
        "latent_vs_no_latent": {
            "agents": ["bw_PAML", "latent_bw_PAML", "vanilla_PAML"],
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
            "agents": ["bw", "bw_PAML", "fw", "fw_PAML"],
            "planning_depths": [1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0]
        }
    },
    "random_linear": {
        "latent_vs_no_latent": {
            "agents": ["bw_PAML", "latent_bw_PAML", "vanilla_PAML"],
            "planning_depths": [1, 1, 0],
            "replay_capacities": [0, 0, 0]
        },
        "bw_fw": {
            "agents": ["bw_PAML", "fw_PAML", "vanilla_PAML"],
            "planning_depths": [1, 1, 0],
            "replay_capacities": [0, 0, 0]
        },
        # "all": {
        #     "agents": ["bw", "mult_bw",
        #                "true_bw", "true_mult_bw",
        #                "fw", "mult_fw",
        #                "fw_PAML", "fw_mult_PAML",
        #                "bw_PAML", "bw_mult_PAML",
        #                "bw_update"],
        #     "planning_depths": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        #     "replay_capacities": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        # },
        "all": {
            "agents": ["bw",
                       "true_bw",
                       "fw",
                       "fw_PAML",
                       "bw_PAML"],
            "planning_depths": [1, 1, 1, 1, 1, ],
            "replay_capacities": [0, 0, 0, 0, 0, 0,]
        },
        "mb_all": {
            "agents": ["mb_bw",
                       "mb_true_bw",
                       "mb_fw",
                       "mb_fw_PAML",
                       "mb_bw_PAML"],
            "planning_depths": [1, 1, 1, 1, 1, ],
            "replay_capacities": [0, 0, 0, 0, 0, ]
        },
        "corr_vs_value_vs_meta": {
            "agents": ["bw_PAML", "bw_vaware", "bw_meta"],
            "planning_depths": [1, 1, 1],
            "replay_capacities": [0, 0, 0]
        },
    },
    "boyan": {
        "latent_vs_no_latent": {
            "agents": ["bw_PAML", "latent_bw_PAML", "vanilla_PAML"],
            "planning_depths": [1, 1, 0],
            "replay_capacities": [0, 0, 0]
        },
        "bw_fw": {
            "agents": ["bw_PAML", "fw_PAML", "vanilla_PAML"],
            "planning_depths": [1, 1, 0],
            "replay_capacities": [0, 0, 0]
        },
        "PAML_vs_MLE_fw_and_bw": {
            "agents": ["bw_PAML", "bw", "fw_PAML", "fw", "vanilla_PAML"],
            "planning_depths": [1, 1, 1, 1, 0],
            "replay_capacities": [0, 0, 0, 0, 0]
        },
        "intr_vs_MLE": {
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
            "agents": ["bw_PAML", "latent_bw_PAML", "vanilla_PAML"],
            "planning_depths": [1, 1, 0],
            "replay_capacities": [0, 0, 0]
        },
        "bw_fw": {
            "agents": ["bw_PAML", "fw_PAML", "vanilla_PAML"],
            "planning_depths": [1, 1, 0],
            "replay_capacities": [0, 0, 0]
        },
        "intr_vs_MLE_fw_and_bw": {
            "agents": ["bw_PAML", "bw", "fw_PAML", "fw", "vanilla_PAML"],
            "planning_depths": [1, 1, 1, 1, 0],
            "replay_capacities": [0, 0, 0, 0, 0]
        },
        "intr_vs_MLE": {
            "agents": ["bw_PAML", "bw"],
            "planning_depths": [1, 1],
            "replay_capacities": [0, 0]
        },
        "corr_vs_value_vs_meta": {
            "agents": ["bw_PAML", "bw_vaware", "bw_meta"],
            "planning_depths": [1, 1, 1],
            "replay_capacities": [0, 0, 0]
        },
    },
    "split": {
        # "all": {
        #     "agents": ["bw", "true_bw", "fw", "true_fw", "bw_recur",
        #                "true_bw_recur"],
        #     "planning_depths": [1, 1, 1, 1, 1, 1],
        #     "replay_capacities": [0, 0, 0, 0, 0, 0]
        # },
        "all_mle": {
            "agents": ["bw_MLE", "true_bw", "fw_MLE",
                       "true_fw", "bw_recur_MLE", "true_bw_recur"],
            "planning_depths": [1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0, 0, 0]
        },
        "bw_c_bw_p_all_mle": {
            "agents": ["c_bw_MLE", "c_true_bw", "p_bw_MLE", "p_true_bw",
                       ],
            "planning_depths": [1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0, 0, 0]
        },
        "bw_p_fw_p_all_mle": {
            "agents": ["p_bw_MLE", "p_true_bw", "p_fw_MLE", "p_true_fw",
                       ],
            "planning_depths": [1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0, 0, 0]
        },
        "fw_c_fw_p_all_mle": {
            "agents": ["c_fw_MLE", "c_true_fw", "p_fw_MLE", "p_true_fw",
                       ],
            "planning_depths": [1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0, 0, 0]
        },
        # "mb_all": {
        #     "agents": ["mb_bw", "mb_true_bw", "mb_fw", "mb_true_fw",
        #                "mb_bw_recur", "mb_true_bw_recur"],
        #     "planning_depths": [1, 1, 1, 1, 1, 1],
        #     "replay_capacities": [0, 0, 0, 0, 0, 0]
        # },
        # "mb_bw_c_fw_p_all_mle": {
        #     "agents": ["mb_c_bw_MLE", "mb_c_true_bw",
        #                "mb_p_fw_MLE", "mb_p_true_fw",
        #                "mb_c_bw_recur_MLE", "mb_c_true_bw_recur"],
        #     "planning_depths": [1, 1, 1, 1, 1, 1],
        #     "replay_capacities": [0, 0, 0, 0, 0, 0]
        # },
        "mb_bw_p_fw_p_all_mle": {
            "agents": ["mb_p_bw_MLE", "mb_p_true_bw",
                       "mb_p_fw_MLE", "mb_p_true_fw"],
                       # "mb_c_bw_recur_MLE", "mb_c_true_bw_recur"],
            "planning_depths": [1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0, 0, 0]
        },
        "mb_bw_c_fw_c_all_mle": {
            "agents": ["mb_c_bw_MLE", "mb_c_true_bw",
                       "mb_c_fw_MLE", "mb_c_true_fw"],
                       # "mb_c_bw_recur_MLE", "mb_c_true_bw_recur"],
            "planning_depths": [1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0, 0, 0]
        },
        "mb_bw_c_all_mle": {
            "agents": ["mb_c_bw_MLE", "mb_c_true_bw"],
            "planning_depths": [1, 1, 1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0, 0, 0, 0, 0]
        },
        "mb_bw_p_all_mle": {
            "agents": ["mb_p_bw_MLE", "mb_p_true_bw"],
            "planning_depths": [1, 1, 1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0, 0, 0, 0, 0]
        },
        "mb_fw_c_all_mle": {
            "agents": ["mb_c_fw_MLE", "mb_c_true_fw"],
            "planning_depths": [1, 1, 1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0, 0, 0, 0, 0]
        },
        "mb_fw_p_all_mle": {
            "agents": ["mb_p_fw_MLE", "mb_p_true_fw"],
            "planning_depths": [1, 1, 1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0, 0, 0, 0, 0]
        },
        "mb_bw_c_bw_p_all_mle": {
            "agents": ["mb_c_bw_MLE", "mb_c_true_bw",
                       "mb_p_bw_MLE", "mb_p_true_bw"],
                       # "mb_c_bw_recur_MLE", "mb_c_true_bw_recur",
                       # "mb_p_bw_recur_MLE", "mb_p_true_bw_recur"],
            "planning_depths": [1, 1, 1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0, 0, 0, 0, 0]
        },
        "mb_fw_c_fw_p_all_mle": {
            "agents": ["mb_c_fw_MLE", "mb_c_true_fw",
                       "mb_p_fw_MLE", "mb_p_true_fw"],
            "planning_depths": [1, 1, 1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0, 0, 0, 0, 0]
        }
    },
    "obstacle": {
        "bw_p_fw_p_all": {
            "agents": ["p_bw", "p_true_bw", "p_fw", "p_fw_PAML", "p_bw_PAML"
                       ],
            "planning_depths": [1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0, 0, 0]
        },
        "all": {
            "agents": ["bw",
                       "true_bw",
                       "fw",
                       "fw_PAML",
                       "bw_PAML"],
            "planning_depths": [1, 1, 1, 1, 1, ],
            "replay_capacities": [0, 0, 0, 0, 0, ]
        },
        "mb_all": {
            "agents": ["mb_bw",
                       "mb_true_bw",
                       "mb_fw",
                       "mb_fw_PAML",
                       "mb_bw_PAML"],
            "planning_depths": [1, 1, 1, 1, 1, ],
            "replay_capacities": [0, 0, 0, 0, 0, ]
        },
    },
    "reg1_obstacle": {
        "all": {
            "agents": ["bw",
                       "true_bw",
                       "fw",
                       "fw_PAML",
                       "bw_PAML"],
            "planning_depths": [1, 1, 1, 1, 1, ],
            "replay_capacities": [0, 0, 0, 0, 0, ]
        },
        "mb_all": {
            "agents": ["mb_bw",
                       "mb_true_bw",
                       "mb_fw",
                       "mb_fw_PAML",
                       "mb_bw_PAML"],
            "planning_depths": [1, 1, 1, 1, 1, ],
            "replay_capacities": [0, 0, 0, 0, 0, ]
        },
    },
    "reg01_obstacle": {
        "all": {
            "agents": ["bw",
                       "true_bw",
                       "fw",
                       "fw_PAML",
                       "bw_PAML"],
            "planning_depths": [1, 1, 1, 1, 1, ],
            "replay_capacities": [0, 0, 0, 0, 0, ]
        },
        "mb_all": {
            "agents": ["mb_bw",
                       "mb_true_bw",
                       "mb_fw",
                       "mb_fw_PAML",
                       "mb_bw_PAML"],
            "planning_depths": [1, 1, 1, 1, 1, ],
            "replay_capacities": [0, 0, 0, 0, 0, ]
        },
    },
    "reg2_obstacle": {
        "all": {
            "agents": ["bw",
                       "true_bw",
                       "fw",
                       "fw_PAML",
                       "bw_PAML"],
            "planning_depths": [1, 1, 1, 1, 1, ],
            "replay_capacities": [0, 0, 0, 0, 0, ]
        },
        "mb_all": {
            "agents": ["mb_bw",
                       "mb_true_bw",
                       "mb_fw",
                       "mb_fw_PAML",
                       "mb_bw_PAML"],
            "planning_depths": [1, 1, 1, 1, 1, ],
            "replay_capacities": [0, 0, 0, 0, 0, ]
        },
    },
}