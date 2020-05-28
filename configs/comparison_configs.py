configs={
    "maze_stoch": {
            "control": {
                "agents": ["p_fw_q",
                           "p_bw_q",
                           "p_true_fw_q"
                           ],
            },
            "pc": {
                "agents": ["p_fw_q",
                           "c_fw_q",
                           "p_true_fw_q",
                           "c_true_fw_q",
                           "p_bw_q",
                           "c_bw_q"]
            }
    },
    "maze_1": {
            "control": {
                "agents": ["p_fw_q",
                           "p_bw_q",
                           "p_true_fw_q"
                           ],
            },
            "pc": {
                "agents": ["p_fw_q",
                           "c_fw_q",
                           "p_true_fw_q",
                           "c_true_fw_q",
                           "p_bw_q",
                           "c_bw_q"]
            }
    },
    "maze_05": {
            "control": {
                "agents": ["p_fw_q",
                           "p_bw_q",
                           "p_true_fw_q"
                           ],
            },
            "pc": {
                "agents": ["p_fw_q",
                           "c_fw_q",
                           "p_true_fw_q",
                           "c_true_fw_q",
                           "p_bw_q",
                           "c_bw_q"]
            }
    },
    "maze_01": {
            "control": {
                "agents": ["p_fw_q",
                           "p_bw_q",
                           "p_bw_q_1",
                           "p_bw_q_2",
                           "p_bw_q_3",
                           "p_true_fw_q"
                           ],
            },
            "pc": {
                "agents": ["p_fw_q",
                           "c_fw_q",
                           "p_true_fw_q",
                           "c_true_fw_q",
                           "p_bw_q",
                            "p_bw_q_1",
                           "c_bw_q"]
            }
    },
    "maze": {
        "control": {
            "agents": ["p_fw_q",
                       "p_bw_q_top_1",
                       "p_bw_q_top_2",
                       "p_bw_q_top_3",
                       "p_bw_q_top_4",
                       "p_bw_q_top_5"
                       ],
        },
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
    "open_medium_maze": {
        "control": {
            "agents": ["p_fw_q",
                       "p_bw_q_top_1",
                       "p_bw_q_top_2",
                       "p_bw_q_top_3",
                       "p_bw_q_top_4",
                       "p_bw_q_top_5"],
        },
    },
    "open_maze": {
        "control": {
            "agents": ["p_fw_q",
                       "p_bw_q_top_1",
                       "p_bw_q_top_2",
                       "p_bw_q_top_3",
                       "p_bw_q_top_4",
                       "p_bw_q_top_5"],
        },
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
        "control": {
            "agents": ["p_fw_q",
                       "p_bw_q_top_1",
                       "p_bw_q_top_2",
                       "p_bw_q_top_3",
                       "p_bw_q_top_4",
                       "p_bw_q_top_5"
                       ],
        },
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
    "large_maze": {
        "control": {
            "agents": [
                "p_fw_q",
                       "p_bw_q_top_1",
                       "p_bw_q_top_2",
                       "p_bw_q_top_3",
                       # "p_bw_q_top_4",
                       # "p_bw_q_top_5"
                       ],
        },
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
                        "p_bw_recur_MLE", "p_true_bw_recur"
                       ],
            "planning_depths": [1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0, 0, 0]
        },
        "bw_c_fw_p_all_mle": {
            "agents": ["c_bw_MLE", "p_bw_MLE", "c_true_bw", "p_fw_MLE", "p_true_fw",
                        "c_bw_recur_MLE", "c_true_bw_recur"
                       ],
            "planning_depths": [1, 1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0, 0, 0, 0]
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
        "mb_bw_c_fw_p_all_mle": {
            "agents": ["mb_c_bw_MLE", "mb_c_true_bw",
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
        "control_all": {
            "agents": [
                       "p_ac_fw",
                       "c_ac_bw",
                       "c_ac_true_bw",
                       "p_ac_fw_PAML",
                       "c_ac_bw_PAML",
                       ],
            "planning_depths": [1, 1, 1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0, 0, 0, 0, 0]
        },
        "bw_p_fw_p_all": {
            "agents": ["p_bw",
                       "p_true_bw",
                       "p_fw",
                       "p_fw_PAML",
                       "p_bw_PAML"
                       ],
            "planning_depths": [1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0, 0, 0]
        },
         "bw_c_fw_p_all": {
            "agents": ["c_bw",
                       "c_true_bw",
                       "c_random_bw",
                       "p_fw",
                       "p_fw_PAML",
                       "c_bw_PAML",
                       "c_bw_random_PAML"
                       ],
             "planning_depths": [1, 1, 1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0, 0, 0, 0, 0]
        },
        "bw_c_bw_p_all": {
            "agents": ["p_bw",
                       "c_bw",
                       "p_true_bw",
                       "c_true_bw",
                       "p_bw_PAML",
                       "c_bw_PAML"
                       ],
            "planning_depths": [1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0, 0, 0]
        },
        "fw_c_fw_p_all": {
            "agents": ["p_fw",
                       "c_fw",
                       "p_fw_PAML",
                       "c_fw_PAML"
                       ],
            "planning_depths": [1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0, 0, 0]
        },
        "mb_bw_c_fw_p_all": {
            "agents": ["mb_c_bw",
                       "mb_c_true_bw",
                       "mb_p_fw",
                       "mb_p_fw_PAML",
                       "mb_c_bw_PAML"
                       ],
            "planning_depths": [1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0, 0, 0]
        },
    },
    "freenorm_obstacle": {
        "bw_p_fw_p_all": {
            "agents": ["p_bw",
                       "p_true_bw",
                       "p_fw",
                       "p_fw_PAML",
                       "p_bw_PAML"
                       ],
            "planning_depths": [1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0, 0, 0]
        },
         "bw_c_fw_p_all": {
            "agents": ["c_bw",
                       "c_true_bw",
                       "c_random_bw",
                       "p_fw",
                       "p_fw_PAML",
                       "c_bw_PAML",
                       "c_bw_random_PAML"
                       ],
             "planning_depths": [1, 1, 1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0, 0, 0, 0, 0]
        },
        "bw_c_bw_p_all": {
            "agents": ["p_bw",
                       "c_bw",
                       "p_true_bw",
                       "c_true_bw",
                       "p_bw_PAML",
                       "c_bw_PAML"
                       ],
            "planning_depths": [1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0, 0, 0]
        },
        "fw_c_fw_p_all": {
            "agents": ["p_fw",
                       "c_fw",
                       "p_fw_PAML",
                       "c_fw_PAML"
                       ],
            "planning_depths": [1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0, 0, 0]
        },
        "mb_bw_c_fw_p_all": {
            "agents": ["mb_c_bw",
                       "mb_c_true_bw",
                       "mb_p_fw",
                       "mb_p_fw_PAML",
                       "mb_c_bw_PAML"
                       ],
            "planning_depths": [1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0, 0, 0]
        },
    },
    "norm05_obstacle": {
        "bw_p_fw_p_all": {
            "agents": ["p_bw",
                       "p_true_bw",
                       "p_fw",
                       "p_fw_PAML",
                       "p_bw_PAML"
                       ],
            "planning_depths": [1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0, 0, 0]
        },
         "bw_c_fw_p_all": {
            "agents": ["c_bw",
                       "c_true_bw",
                       "c_random_bw",
                       "p_fw",
                       "p_fw_PAML",
                       "c_bw_PAML",
                       "c_bw_random_PAML"
                       ],
             "planning_depths": [1, 1, 1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0, 0, 0, 0, 0]
        },
        "bw_c_bw_p_all": {
            "agents": ["p_bw",
                       "c_bw",
                       "p_true_bw",
                       "c_true_bw",
                       "p_bw_PAML",
                       "c_bw_PAML"
                       ],
            "planning_depths": [1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0, 0, 0]
        },
        "fw_c_fw_p_all": {
            "agents": ["p_fw",
                       "c_fw",
                       "p_fw_PAML",
                       "c_fw_PAML"
                       ],
            "planning_depths": [1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0, 0, 0]
        },
        "mb_bw_c_fw_p_all": {
            "agents": ["mb_c_bw",
                       "mb_c_true_bw",
                       "mb_p_fw",
                       "mb_p_fw_PAML",
                       "mb_c_bw_PAML"
                       ],
            "planning_depths": [1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0, 0, 0]
        },
    },
    "norm01_obstacle": {
        "bw_p_fw_p_all": {
            "agents": ["p_bw",
                       "p_true_bw",
                       "p_fw",
                       "p_fw_PAML",
                       "p_bw_PAML"
                       ],
            "planning_depths": [1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0, 0, 0]
        },
        "bw_c_fw_p_all": {
            "agents": ["c_bw",
                       "c_true_bw",
                       "c_random_bw",
                       "p_fw",
                       "p_fw_PAML",
                       "c_bw_PAML",
                       "c_bw_random_PAML"
                       ],
            "planning_depths": [1, 1, 1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0, 0, 0, 0, 0]
        },
        "bw_c_bw_p_all": {
            "agents": ["p_bw",
                       "c_bw",
                       "p_true_bw",
                       "c_true_bw",
                       "p_bw_PAML",
                       "c_bw_PAML"
                       ],
            "planning_depths": [1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0, 0, 0]
        },
        "fw_c_fw_p_all": {
            "agents": ["p_fw",
                       "c_fw",
                       "p_fw_PAML",
                       "c_fw_PAML"
                       ],
            "planning_depths": [1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0, 0, 0]
        },
        "mb_bw_c_fw_p_all": {
            "agents": ["mb_c_bw",
                       "mb_c_true_bw",
                       "mb_p_fw",
                       "mb_p_fw_PAML",
                       "mb_c_bw_PAML"
                       ],
            "planning_depths": [1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0, 0, 0]
        },
    },
    "norm1_obstacle": {
        "bw_p_fw_p_all": {
            "agents": ["p_bw",
                       "p_true_bw",
                       "p_random_bw",
                       "p_fw",
                       "p_fw_PAML",
                       "p_bw_PAML"
                       "p_bw_random_PAML"
                       ],
            "planning_depths": [1, 1, 1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0, 0, 0, 0, 0]
        },
        "bw_c_fw_p_all": {
            "agents": ["c_bw",
                       "c_true_bw",
                       "c_random_bw",
                       "p_fw",
                       "p_fw_PAML",
                       "c_bw_PAML",
                       "c_bw_random_PAML"
                       ],
             "planning_depths": [1, 1, 1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0, 0, 0, 0, 0]
        },
        "bw_c_bw_p_all": {
            "agents": ["p_bw",
                       "c_bw",
                       "p_true_bw",
                       "c_true_bw",
                       "p_bw_PAML",
                       "c_bw_PAML"
                       ],
            "planning_depths": [1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0, 0, 0]
        },
        "fw_c_fw_p_all": {
            "agents": ["p_fw",
                       "c_fw",
                       "p_fw_PAML",
                       "c_fw_PAML"
                       ],
            "planning_depths": [1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0, 0, 0]
        },
        "mb_bw_c_fw_p_all": {
            "agents": ["mb_c_bw",
                       "mb_c_true_bw",
                       "mb_p_fw",
                       "mb_p_fw_PAML",
                       "mb_c_bw_PAML"
                       ],
            "planning_depths": [1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0, 0, 0]
        },
    },
    "norm0_obstacle": {
        "bw_p_fw_p_all": {
            "agents": ["p_bw",
                       "p_true_bw",
                       "p_random_bw",
                       "p_fw",
                       "p_fw_PAML",
                       "p_bw_PAML"
                       "p_bw_random_PAML"
                       ],
            "planning_depths": [1, 1, 1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0, 0, 0, 0, 0]
        },
        "bw_c_fw_p_all": {
            "agents": ["c_bw",
                       "c_true_bw",
                       "c_random_bw",
                       "p_fw",
                       "p_fw_PAML",
                       "c_bw_PAML",
                       "c_bw_random_PAML"
                       ],
             "planning_depths": [1, 1, 1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0, 0, 0, 0, 0]
        },
        "bw_c_bw_p_all": {
            "agents": ["p_bw",
                       "c_bw",
                       "p_true_bw",
                       "c_true_bw",
                       "p_bw_PAML",
                       "c_bw_PAML"
                       ],
            "planning_depths": [1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0, 0, 0]
        },
        "fw_c_fw_p_all": {
            "agents": ["p_fw",
                       "c_fw",
                       "p_fw_PAML",
                       "c_fw_PAML"
                       ],
            "planning_depths": [1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0, 0, 0]
        },
        "mb_bw_c_fw_p_all": {
            "agents": ["mb_c_bw",
                       "mb_c_true_bw",
                       "mb_p_fw",
                       "mb_p_fw_PAML",
                       "mb_c_bw_PAML"
                       ],
            "planning_depths": [1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0, 0, 0]
        },
    },
    "bipartite": {
        "MLE_PAML_all_mle": {
            "agents": ["c_bw_MLE",
                        "c_bw_proj_MLE",
                       "p_fw_MLE",
                       "c_bw_PAML",
                       "p_fw_PAML"
                       ],
            "planning_depths": [1, 1, 1, 1, 1, 1],
            "max_norms": [0, 0.01, 0, 0, 0, 0],
            "replay_capacities": [0, 0, 0, 0, 0, 0]
        },
        "bw_c_fw_p_all_mle": {
            "agents": ["c_bw_MLE",
                       "p_bw_MLE",
                       "c_true_bw",
                       "p_fw_MLE",
                       "p_true_fw",
                       ],
            "planning_depths": [1, 1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0, 0, 0, 0]
        },
        # "mb_bwfw_c_fw_p_all_mle": {
        #     "agents": ["mb_c_bw_MLE", "mb_c_true_bw",
        #                "mb_c_bwfw_MLE", "mb_c_true_bwfw",
        #                "mb_p_fw_MLE", "mb_p_true_fw",
        #                ],
        #     "planning_depths": [1, 1, 1, 1, 1, 1],
        #     "replay_capacities": [0, 0, 0, 0, 0, 0]
        # },
        "mb_corr_all_mle": {
            "agents": ["mb_c_bw_MLE",
                       "mb_c_bwfw_MLE",
                       "mb_c_true_bw",
                       "mb_c_true_bwfw",
                       ],
            # "mb_c_bw_recur_MLE", "mb_c_true_bw_recur"],
            "planning_depths": [1, 1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0, 0, 0, 0]
        },
        "mb_bw_c_fw_p_all_mle": {
            "agents": ["mb_c_bw_MLE",
                        # "mb_c_bwfw_MLE",
                       "mb_c_random_bw",
                       "mb_c_true_bw",
                       # "mb_c_true_bwfw",
                       "mb_p_fw_MLE",
                       "mb_p_true_fw",],
                       # "mb_c_bw_recur_MLE", "mb_c_true_bw_recur"],
            "planning_depths": [1, 1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0, 0, 0, 0]
        },
        "mb_MLE_PAML_all_mle": {
            "agents": ["mb_c_bw_MLE",
                        "mb_c_random_bw",
                        "mb_c_bw_proj_MLE",
                       "mb_c_bw_PAML", ],
                       # "mb_c_bw_recur_MLE", "mb_c_true_bw_recur"],
            "planning_depths": [1, 1, 1, 1, 1, 1],
            # "max_norms": [0, 0.01, 0, 0, 0, 0],
            "replay_capacities": [0, 0, 0, 0, 0, 0]
        },
    },
    "fanin": {
        "mb_corr_all_mle": {
            "agents": ["mb_c_bw_MLE",
                       "mb_c_bwfw_MLE",
                       "mb_c_true_bw",
                       "mb_c_true_bwfw",
                       ],
            # "mb_c_bw_recur_MLE", "mb_c_true_bw_recur"],
            "planning_depths": [1, 1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0, 0, 0, 0]
        },
        "mb_MLE_PAML_all_mle": {
            "agents": ["mb_c_bw_MLE",
                       "mb_c_bw_proj_MLE",
                       "mb_c_bw_PAML", ],
            # "mb_c_bw_recur_MLE", "mb_c_true_bw_recur"],
            "planning_depths": [1, 1, 1, 1, 1, 1],
            "max_norms": [0, 0.1, 0, 0, 0, 0],
            "replay_capacities": [0, 0, 0, 0, 0, 0]
        },
        "bw_c_fw_p_all_mle": {
            "agents": ["c_bw_MLE",
                       "p_bw_MLE",
                       "c_true_bw",
                       "p_fw_MLE",
                       "p_true_fw",
                       ],
            "planning_depths": [1, 1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0, 0, 0, 0]
        },
        "bw_p_fw_p_all_mle": {
            "agents": ["p_bw_MLE", "p_true_bw", "p_fw_MLE", "p_true_fw",
                       "p_bw_recur_MLE", "p_true_bw_recur"
                       ],
            "planning_depths": [1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0, 0, 0]
        },
        "mb_bw_c_fw_p_all_mle": {
            "agents": ["mb_c_bw_MLE", "mb_c_true_bw",
                       "mb_c_random_bw",
                       "mb_p_fw_MLE", "mb_p_true_fw"],
            "planning_depths": [1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0, 0, 0]
        },
        "mb_bwfw_c_fw_p_all_mle": {
            "agents": ["mb_c_bwfw_MLE", "mb_c_true_bwfw",
                       "mb_p_fw_MLE", "mb_p_true_fw",
                       ],
            "planning_depths": [1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0, 0, 0]
        },
        "mb_MLE_PAML_all_mle": {
            "agents": ["mb_c_bw_MLE",
                       "mb_c_random_bw",
                       "mb_c_bw_proj_MLE",
                       "mb_c_bw_PAML", ],
            # "mb_c_bw_recur_MLE", "mb_c_true_bw_recur"],
            "planning_depths": [1, 1, 1, 1, 1, 1],
            # "max_norms": [0, 0.1, 0, 0, 0, 0],
            "replay_capacities": [0, 0, 0, 0, 0, 0]
        },
    },
"bipartite_10_1": {
    "mb_corr_all_mle": {
        "agents": ["mb_c_bw_MLE",
                   "mb_c_bwfw_MLE",
                   "mb_c_true_bw",
                   "mb_c_true_bwfw",
                   ],
        # "mb_c_bw_recur_MLE", "mb_c_true_bw_recur"],
        "planning_depths": [1, 1, 1, 1, 1, 1, 1],
        "replay_capacities": [0, 0, 0, 0, 0, 0, 0]
    },
    "mb_MLE_PAML_all_mle": {
        "agents": ["mb_c_bw_MLE",
                   "mb_c_bw_proj_MLE",
                   "mb_c_bw_PAML", ],
        # "mb_c_bw_recur_MLE", "mb_c_true_bw_recur"],
        "planning_depths": [1, 1, 1, 1, 1, 1],
        "max_norms": [0, 0.1, 0, 0, 0, 0],
        "replay_capacities": [0, 0, 0, 0, 0, 0]
    },
        "bw_c_fw_p_all_mle": {
            "agents": ["c_bw_MLE",
                       "p_bw_MLE",
                       "c_true_bw",
                       "p_fw_MLE",
                       "p_true_fw",
                       ],
            "planning_depths": [1, 1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0, 0, 0, 0]
        },
        "bw_p_fw_p_all_mle": {
            "agents": ["p_bw_MLE", "p_true_bw", "p_fw_MLE", "p_true_fw",
                        "p_bw_recur_MLE", "p_true_bw_recur"
                       ],
            "planning_depths": [1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0, 0, 0]
        },
        "mb_bw_c_fw_p_all_mle": {
            "agents": ["mb_c_bw_MLE", "mb_c_true_bw", "mb_c_random_bw",
                       "mb_p_fw_MLE", "mb_p_true_fw"],
            "planning_depths": [1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0, 0, 0]
        },
        "mb_bwfw_c_fw_p_all_mle": {
            "agents": ["mb_c_bwfw_MLE", "mb_c_true_bwfw",
                       "mb_p_fw_MLE", "mb_p_true_fw",
                       ],
            "planning_depths": [1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0, 0, 0]
        },
    },
"bipartite_100_1": {
    "mb_corr_all_mle": {
        "agents": ["mb_c_bw_MLE",
                   "mb_c_bwfw_MLE",
                   "mb_c_true_bw",
                   "mb_c_true_bwfw",
                   ],
        # "mb_c_bw_recur_MLE", "mb_c_true_bw_recur"],
        "planning_depths": [1, 1, 1, 1, 1, 1, 1],
        "replay_capacities": [0, 0, 0, 0, 0, 0, 0]
    },
        "bw_c_fw_p_all_mle": {
            "agents": ["c_bw_MLE",
                       "p_bw_MLE",
                       "c_true_bw",
                       "p_fw_MLE",
                       "p_true_fw",
                       ],
            "planning_depths": [1, 1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0, 0, 0, 0]
        },
        "mb_bw_c_fw_p_all_mle": {
            "agents": ["mb_c_bw_MLE", "mb_c_random_bw", "mb_c_true_bw",
                       "mb_p_fw_MLE", "mb_p_true_fw",],
                       # "mb_c_bw_recur_MLE", "mb_c_true_bw_recur"],
            "planning_depths": [1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0, 0, 0]
        },
        "mb_bwfw_c_fw_p_all_mle": {
            "agents": ["mb_c_bwfw_MLE", "mb_c_true_bwfw",
                       "mb_p_fw_MLE", "mb_p_true_fw",
                       ],
            "planning_depths": [1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0, 0, 0]
        },
    },
"bipartite_1_10": {
    "mb_corr_all_mle": {
        "agents": ["mb_c_bw_MLE",
                   "mb_c_bwfw_MLE",
                   "mb_c_true_bw",
                   "mb_c_true_bwfw",
                   ],
        # "mb_c_bw_recur_MLE", "mb_c_true_bw_recur"],
        "planning_depths": [1, 1, 1, 1, 1, 1, 1],
        "replay_capacities": [0, 0, 0, 0, 0, 0, 0]
    },
        "bw_c_fw_p_all_mle": {
            "agents": ["c_bw_MLE",
                       "p_bw_MLE",
                       "c_true_bw",
                       "p_fw_MLE",
                       "p_true_fw",
                       ],
            "planning_depths": [1, 1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0, 0, 0, 0]
        },
        "bw_p_fw_p_all_mle": {
            "agents": ["p_bw_MLE", "p_true_bw", "p_fw_MLE", "p_true_fw",
                        "p_bw_recur_MLE", "p_true_bw_recur"
                       ],
            "planning_depths": [1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0, 0, 0]
        },
        "mb_bw_c_fw_p_all_mle": {
            "agents": ["mb_c_bw_MLE", "mb_c_random_bw", "mb_c_true_bw",
                       "mb_p_fw_MLE", "mb_p_true_fw",],
                       # "mb_c_bw_recur_MLE", "mb_c_true_bw_recur"],
            "planning_depths": [1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0, 0, 0]
        },
        "mb_bwfw_c_fw_p_all_mle": {
            "agents": ["mb_c_bwfw_MLE", "mb_c_true_bwfw",
                       "mb_p_fw_MLE", "mb_p_true_fw",
                       ],
            "planning_depths": [1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0, 0, 0]
        },
    },
"bipartite_1_100": {
    "mb_corr_all_mle": {
        "agents": ["mb_c_bw_MLE",
                   "mb_c_bwfw_MLE",
                   "mb_c_true_bw",
                   "mb_c_true_bwfw",
                   ],
        # "mb_c_bw_recur_MLE", "mb_c_true_bw_recur"],
        "planning_depths": [1, 1, 1, 1, 1, 1, 1],
        "replay_capacities": [0, 0, 0, 0, 0, 0, 0]
    },
        "bw_c_fw_p_all_mle": {
            "agents": ["c_bw_MLE",
                       "p_bw_MLE",
                       "c_true_bw",
                       "p_fw_MLE",
                       "p_true_fw",
                       ],
            "planning_depths": [1, 1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0, 0, 0, 0]
        },
        "bw_p_fw_p_all_mle": {
            "agents": ["p_bw_MLE", "p_true_bw", "p_fw_MLE", "p_true_fw",
                        "p_bw_recur_MLE", "p_true_bw_recur"
                       ],
            "planning_depths": [1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0, 0, 0]
        },
    "mb_bw_c_fw_p_all_mle": {
        "agents": ["mb_c_bw_MLE", "mb_c_random_bw", "mb_c_true_bw",
                   "mb_p_fw_MLE", "mb_p_true_fw", ],
        # "mb_c_bw_recur_MLE", "mb_c_true_bw_recur"],
        "planning_depths": [1, 1, 1, 1, 1, 1],
        "replay_capacities": [0, 0, 0, 0, 0, 0]
    },
        "mb_bwfw_c_fw_p_all_mle": {
            "agents": ["mb_c_bwfw_MLE", "mb_c_true_bwfw",
                       "mb_p_fw_MLE", "mb_p_true_fw",
                       ],
            "planning_depths": [1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0, 0, 0]
        },
    },
"bipartite_100_10_1_2L": {
    "mb_corr_all_mle": {
        "agents": ["mb_c_bw_MLE",
                   "mb_c_bwfw_MLE",
                   "mb_c_true_bw",
                   "mb_c_true_bwfw",
                   ],
        # "mb_c_bw_recur_MLE", "mb_c_true_bw_recur"],
        "planning_depths": [1, 1, 1, 1, 1, 1, 1],
        "replay_capacities": [0, 0, 0, 0, 0, 0, 0]
    },
    "bw_c_fw_p_all_mle": {
        "agents": ["c_bw_MLE",
                   "p_bw_MLE",
                   "c_true_bw",
                   "p_true_bw",
                   "p_fw_MLE",
                   "c_fw_MLE",
                   "p_true_fw",
                   "c_true_fw",
                   ],
        "planning_depths": [1, 1, 1, 1, 1, 1, 1],
        "replay_capacities": [0, 0, 0, 0, 0, 0, 0]
    },
     "mb_pc_all_mle": {
        "agents": [
            "mb_c_bw_MLE",
            "mb_p_bw_MLE",
            "mb_c_true_bw",
            "mb_p_true_bw",
            "mb_p_fw_MLE",
            "mb_c_fw_MLE",
            "mb_p_true_fw",
            "mb_c_true_fw",
        ],
        # "mb_c_bw_recur_MLE", "mb_c_true_bw_recur"],
        "planning_depths": [1, 1, 1, 1, 1, 1],
        "replay_capacities": [0, 0, 0, 0, 0, 0]
    },
    "mb_bw_c_fw_p_all_mle": {
        "agents": [
            "mb_c_bw_MLE",
            "mb_c_random_bw",
            "mb_c_true_bw",
            "mb_p_fw_MLE",
            "mb_p_true_fw",
        ],
        # "mb_c_bw_recur_MLE", "mb_c_true_bw_recur"],
        "planning_depths": [1, 1, 1, 1, 1, 1],
        "replay_capacities": [0, 0, 0, 0, 0, 0]
    },
    "MLE_PAML_all_mle": {
        "agents": [
            # "c_bw_MLE",
            "p_bw_MLE",
            "p_bw_proj_MLE",
            "p_bw_proj_MLE",
            # "p_fw_MLE",
            # "c_bw_PAML",
            "p_bw_PAML",
            "p_bw_proj_PAML",
            # "p_fw_PAML"
        ],
        "planning_depths": [1, 1, 1, 1, 1, 1],
        "max_norms": [0, 1.0, 0.1, 0, 0, 0],
        "replay_capacities": [0, 0, 0, 0, 0, 0]
    },
    },
"bipartite_5L": {
    "mb_corr_all_mle": {
        "agents": ["mb_c_bw_MLE",
                   "mb_c_bwfw_MLE",
                   "mb_c_true_bw",
                   "mb_c_true_bwfw",
                   ],
        # "mb_c_bw_recur_MLE", "mb_c_true_bw_recur"],
        "planning_depths": [1, 1, 1, 1, 1, 1, 1],
        "replay_capacities": [0, 0, 0, 0, 0, 0, 0]
    },
    "bw_c_fw_p_all_mle": {
        "agents": ["c_bw_MLE",
                   "p_bw_MLE",
                   "c_true_bw",
                   "p_true_bw",
                   "p_fw_MLE",
                   "c_fw_MLE",
                   "p_true_fw",
                   "c_true_fw",
                   ],
        "planning_depths": [1, 1, 1, 1, 1, 1, 1],
        "replay_capacities": [0, 0, 0, 0, 0, 0, 0]
    },
    "mb_pc_all_mle": {
        "agents": [
            "mb_c_bw_MLE",
            "mb_p_bw_MLE",
            "mb_c_true_bw",
            "mb_p_true_bw",
            "mb_p_fw_MLE",
            "mb_c_fw_MLE",
            "mb_p_true_fw",
            "mb_c_true_fw",
        ],
        # "mb_c_bw_recur_MLE", "mb_c_true_bw_recur"],
        "planning_depths": [1, 1, 1, 1, 1, 1],
        "replay_capacities": [0, 0, 0, 0, 0, 0]
    },
    "mb_bw_c_fw_p_all_mle": {
        "agents": [
            "mb_c_bw_MLE",
            "mb_c_random_bw",
            "mb_c_true_bw",
            "mb_p_fw_MLE",
            "mb_p_true_fw",
        ],
        # "mb_c_bw_recur_MLE", "mb_c_true_bw_recur"],
        "planning_depths": [1, 1, 1, 1, 1, 1],
        "replay_capacities": [0, 0, 0, 0, 0, 0]
    },
        "MLE_PAML_all_mle": {
            "agents": [
                       # "c_bw_MLE",
                       "p_bw_MLE",
                       "p_bw_proj_MLE",
                       "p_bw_proj_MLE",
                       # "p_fw_MLE",
                       # "c_bw_PAML",
                       "p_bw_PAML",
                       "p_bw_proj_PAML",
                       # "p_fw_PAML"
                       ],
            "planning_depths": [1, 1, 1, 1, 1, 1],
            "max_norms": [0, 1.0, 0.1, 0, 0, 0],
            "replay_capacities": [0, 0, 0, 0, 0, 0]
            },
    },
    "bipartite_linear": {
        "control_all": {
            "agents": [
                       "p_ac_fw",
                       "c_ac_bw",
                       "c_ac_true_bw",
                       "p_ac_fw_PAML",
                       "c_ac_bw_PAML",
                       ],
            "planning_depths": [1, 1, 1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0, 0, 0, 0, 0]
        },
        "bw_p_fw_p_all": {
            "agents": ["p_bw",
                       "p_true_bw",
                       "p_fw",
                       "p_fw_PAML",
                       "p_bw_PAML"
                       ],
            "planning_depths": [1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0, 0, 0]
        },
         "bw_c_fw_p_all": {
            "agents": ["c_bw",
                       # "c_true_bw",
                       # "c_random_bw",
                       "p_fw",
                       "p_fw_PAML",
                       "c_bw_PAML",
                       # "c_bw_random_PAML"
                       ],
             "planning_depths": [1, 1, 1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0, 0, 0, 0, 0]
        },
        "bw_c_bw_p_all": {
            "agents": ["p_bw",
                       "c_bw",
                       "p_true_bw",
                       "c_true_bw",
                       "p_bw_PAML",
                       "c_bw_PAML"
                       ],
            "planning_depths": [1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0, 0, 0]
        },
        "fw_c_fw_p_all": {
            "agents": ["p_fw",
                       "c_fw",
                       "p_fw_PAML",
                       "c_fw_PAML"
                       ],
            "planning_depths": [1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0, 0, 0]
        },
        "mb_bw_c_fw_p_all": {
            "agents": ["mb_c_bw",
                       "mb_c_true_bw",
                       "mb_p_fw",
                       "mb_p_fw_PAML",
                       "mb_c_bw_PAML"
                       ],
            "planning_depths": [1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0, 0, 0]
        },
    },
    "wide_linear": {
        "control_all": {
            "agents": [
                       "p_ac_fw",
                       "c_ac_bw",
                       "c_ac_true_bw",
                       "p_ac_fw_PAML",
                       "c_ac_bw_PAML",
                       ],
            "planning_depths": [1, 1, 1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0, 0, 0, 0, 0]
        },
        "bw_p_fw_p_all": {
            "agents": ["p_bw",
                       "p_true_bw",
                       "p_fw",
                       "p_fw_PAML",
                       "p_bw_PAML"
                       ],
            "planning_depths": [1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0, 0, 0]
        },
         "bw_c_fw_p_all": {
            "agents": ["c_bw",
                       # "c_true_bw",
                       # "c_random_bw",
                       "p_fw",
                       "p_fw_PAML",
                       "c_bw_PAML",
                       # "c_bw_random_PAML"
                       ],
             "planning_depths": [1, 1, 1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0, 0, 0, 0, 0]
        },
        "bw_c_bw_p_all": {
            "agents": ["p_bw",
                       "c_bw",
                       "p_true_bw",
                       "c_true_bw",
                       "p_bw_PAML",
                       "c_bw_PAML"
                       ],
            "planning_depths": [1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0, 0, 0]
        },
        "fw_c_fw_p_all": {
            "agents": ["p_fw",
                       "c_fw",
                       "p_fw_PAML",
                       "c_fw_PAML"
                       ],
            "planning_depths": [1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0, 0, 0]
        },
        "mb_bw_c_fw_p_all": {
            "agents": ["mb_c_bw",
                       "mb_c_true_bw",
                       "mb_p_fw",
                       "mb_p_fw_PAML",
                       "mb_c_bw_PAML"
                       ],
            "planning_depths": [1, 1, 1, 1, 1, 1],
            "replay_capacities": [0, 0, 0, 0, 0, 0]
        },
    },
}
