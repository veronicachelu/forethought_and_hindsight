config = {
    "vanilla": {
        "task_type": "prediction",
        "control_agent": "q",
        "pg": False,
        "run_mode": "vanilla",
        "planning_depth": 0,
        "planning_iter": 1,
        "latent": False,
        "num_hidden_layers": 0,
        "num_units": 0,
        "model_family": "MLE",
        "target_networks": False,
        "class": {"linear": "LpVanilla",
                  "tabular": "TpVanilla"}
    },

    #### BW RECURRENT ###
    "p_bw_recur": {
        "task_type": "prediction",
        "control_agent": "q",
        "pg": False,
        "run_mode": "p_bw_recur",
        "planning_iter": 1,
        "latent": False,
        "num_hidden_layers": 0,
        "num_units": 0,
        "model_family": "MLE",
        "target_networks": False,
        "class": {"linear": "LpBwRecur",
                  "tabular": "TpBwRecur"},
    },
    "p_true_bw_recur": {
        "task_type": "prediction",
        "control_agent": "q",
        "pg": False,
        "run_mode": "p_true_bw_recur",
        "planning_iter": 1,
        "latent": False,
        "num_hidden_layers": 0,
        "num_units": 0,
        "model_family": "true",
        "target_networks": False,
        "class": {"linear": "LpBwRecur",
                  "tabular": "TpTrueBwRecur"},
    },
    "c_bw_recur": {
        "task_type": "prediction",
        "control_agent": "q",
        "pg": False,
        "run_mode": "c_bw_recur",
        "planning_iter": 1,
        "latent": False,
        "num_hidden_layers": 0,
        "num_units": 0,
        "model_family": "MLE",
        "target_networks": False,
        "class": {"linear": "LpBwRecur",
                  "tabular": "TpBwRecur"},
    },
    "c_true_bw_recur": {
        "task_type": "prediction",
        "control_agent": "q",
        "pg": False,
        "run_mode": "c_true_bw_recur",
        "planning_iter": 1,
        "latent": False,
        "num_hidden_layers": 0,
        "num_units": 0,
        "model_family": "true",
        "target_networks": False,
        "class": {"linear": "LpBwRecur",
                  "tabular": "TpTrueBwRecur"},
    },
    #### BW REC MB ####
    "mb_c_bw_recur": {
        "task_type": "prediction",
        "control_agent": "q",
        "pg": False,
        "run_mode": "mb_c_bw_recur",
        "planning_iter": 1,
        "latent": False,
        "num_hidden_layers": 0,
        "num_units": 0,
        "model_family": "MLE",
        "target_networks": False,
        "class": {"linear": "LpBwRecur",
                  "tabular": "TpBwRecur"},
    },
    "mb_p_bw_recur": {
        "task_type": "prediction",
        "control_agent": "q",
        "pg": False,
        "run_mode": "mb_p_bw_recur",
        "planning_iter": 1,
        "latent": False,
        "num_hidden_layers": 0,
        "num_units": 0,
        "model_family": "MLE",
        "target_networks": False,
        "class": {"linear": "LpBwRecur",
                  "tabular": "TpBwRecur"},
    },
    "mb_c_true_bw_recur": {
        "task_type": "prediction",
        "control_agent": "q",
        "pg": False,
        "run_mode": "mb_c_true_bw_recur",
        "planning_iter": 1,
        "latent": False,
        "num_hidden_layers": 0,
        "num_units": 0,
        "model_family": "true",
        "target_networks": False,
        "class": {"linear": "LpBwRecur",
                  "tabular": "TpTrueBwRecur"},
    },
    "mb_p_true_bw_recur": {
        "task_type": "prediction",
        "control_agent": "q",
        "pg": False,
        "run_mode": "mb_p_true_bw_recur",
        "planning_iter": 1,
        "latent": False,
        "num_hidden_layers": 0,
        "num_units": 0,
        "model_family": "true",
        "target_networks": False,
        "class": {"linear": "LpBwRecur",
                  "tabular": "TpTrueBwRecur"},
    },
    ### BW REC MLE ####
    "p_bw_recur_MLE": {
        "task_type": "prediction",
        "control_agent": "q",
        "pg": False,
        "run_mode": "p_bw_recur_mle",
        "planning_iter": 1,
        "latent": False,
        "num_hidden_layers": 0,
        "num_units": 0,
        "model_family": "MLE",
        "target_networks": False,
        "class": {"linear": "LpBwRecur",
                  "tabular": "TpBwRecurMLE"},
    },
    "c_bw_recur_MLE": {
        "task_type": "prediction",
        "control_agent": "q",
        "pg": False,
        "run_mode": "c_bw_recur_mle",
        "planning_iter": 1,
        "latent": False,
        "num_hidden_layers": 0,
        "num_units": 0,
        "model_family": "MLE",
        "target_networks": False,
        "class": {"linear": "LpBwRecur",
                  "tabular": "TpBwRecurMLE"},
    },
    #### BW REC MB MLE ###
    "mb_p_bw_recur_MLE": {
        "task_type": "prediction",
        "control_agent": "q",
        "pg": False,
        "run_mode": "mb_p_bw_recur_mle",
        "planning_iter": 1,
        "latent": False,
        "num_hidden_layers": 0,
        "num_units": 0,
        "model_family": "MLE",
        "target_networks": False,
        "class": {"linear": "LpBwRecur",
                  "tabular": "TpBwRecurMLE"},
    },
    "mb_c_bw_recur_MLE": {
        "task_type": "prediction",
        "control_agent": "q",
        "pg": False,
        "run_mode": "mb_c_bw_recur_mle",
        "planning_iter": 1,
        "latent": False,
        "num_hidden_layers": 0,
        "num_units": 0,
        "model_family": "MLE",
        "target_networks": False,
        "class": {"linear": "LpBwRecur",
                  "tabular": "TpBwRecurMLE"},
    },

    #### BW ####
    "p_bw": {
        "task_type": "prediction",
        "control_agent": "q",
        "pg": False,
        "run_mode": "p_bw",
        "planning_iter": 1,
        "latent": False,
        "num_hidden_layers": 0,
        "num_units": 0,
        "model_family": "MLE",
        "target_networks": False,
        "class": {"linear": "LpBw",
                  "tabular": "TpBw"},
    },
    "p_true_bw": {
        "task_type": "prediction",
        "control_agent": "q",
        "pg": False,
        "run_mode": "p_true_bw",
        "planning_iter": 1,
        "latent": False,
        "num_hidden_layers": 0,
        "num_units": 0,
        "model_family": "true",
        "target_networks": False,
        "class": {"linear": "LpTrueBw",
                  "tabular": "TpTrueBw"},
    },
    "c_bw": {
        "task_type": "prediction",
        "control_agent": "q",
        "pg": False,
        "run_mode": "c_bw",
        "planning_iter": 1,
        "latent": False,
        "num_hidden_layers": 0,
        "num_units": 0,
        "model_family": "MLE",
        "target_networks": False,
        "class": {"linear": "LpBw",
                  "tabular": "TpBw"},
    },
    "c_true_bw": {
        "task_type": "prediction",
        "control_agent": "q",
        "pg": False,
        "run_mode": "c_true_bw",
        "planning_iter": 1,
        "latent": False,
        "num_hidden_layers": 0,
        "num_units": 0,
        "model_family": "true",
        "target_networks": False,
        "class": {"linear": "LpTrueBw",
                  "tabular": "TpTrueBw"},
    },
    #### BW MLE ###
    "p_bw_MLE": {
        "task_type": "prediction",
        "control_agent": "q",
        "pg": False,
        "run_mode": "p_bw_mle",
        "planning_iter": 1,
        "latent": False,
        "num_hidden_layers": 0,
        "num_units": 0,
        "model_family": "MLE",
        "target_networks": False,
        "class": {"linear": "LpBw",
                  "tabular": "TpBwMLE"},
    },
    "p_bw_proj_MLE": {
        "task_type": "prediction",
        "control_agent": "q",
        "pg": False,
        "run_mode": "p_bw_proj_mle",
        "planning_iter": 1,
        "latent": False,
        "num_hidden_layers": 0,
        "num_units": 0,
        "model_family": "MLE",
        "target_networks": False,
        "class": {"linear": "LpBw",
                  "tabular": "TpBwProjMLE"},
    },
    "c_bw_MLE": {
        "task_type": "prediction",
        "control_agent": "q",
        "pg": False,
        "run_mode": "c_bw_mle",
        "planning_iter": 1,
        "latent": False,
        "num_hidden_layers": 0,
        "num_units": 0,
        "model_family": "MLE",
        "target_networks": False,
        "class": {"linear": "LpBw",
                  "tabular": "TpBwMLE"},
    },
    "c_bw_proj_MLE": {
        "task_type": "prediction",
        "control_agent": "q",
        "pg": False,
        "run_mode": "c_bw_proj_mle",
        "planning_iter": 1,
        "latent": False,
        "num_hidden_layers": 0,
        "num_units": 0,
        "model_family": "MLE",
        "target_networks": False,
        "class": {"linear": "LpBw",
                  "tabular": "TpBwProjMLE"},
    },

    #### BW MB ####
    "mb_c_bw_MLE": {
        "task_type": "prediction",
        "control_agent": "q",
        "pg": False,
        "run_mode": "mb_c_bw_mle",
        "planning_iter": 1,
        "latent": False,
        "num_hidden_layers": 0,
        "num_units": 0,
        "model_family": "MLE",
        "target_networks": False,
        "class": {"linear": "LpBw",
                  "tabular": "TpBwMLE"},
    },
    "mb_c_bw_proj_MLE": {
        "task_type": "prediction",
        "control_agent": "q",
        "pg": False,
        "run_mode": "mb_c_bw_proj_mle",
        "planning_iter": 1,
        "latent": False,
        "num_hidden_layers": 0,
        "num_units": 0,
        "model_family": "MLE",
        "target_networks": False,
        "class": {"linear": "LpBw",
                  "tabular": "TpBwProjMLE"},
    },
    "mb_p_bw_MLE":{
        "task_type": "prediction",
        "control_agent": "q",
        "pg": False,
        "run_mode": "mb_p_bw_mle",
        "planning_iter": 1,
        "latent": False,
        "num_hidden_layers": 0,
        "num_units": 0,
        "model_family": "MLE",
        "target_networks": False,
        "class": {"linear": "LpBw",
                  "tabular": "TpBwMLE"},
    },
    "mb_c_bw": {
        "task_type": "prediction",
        "control_agent": "q",
        "pg": False,
        "run_mode": "mb_c_bw",
        "planning_iter": 1,
        "latent": False,
        "num_hidden_layers": 0,
        "num_units": 0,
        "model_family": "MLE",
        "target_networks": False,
        "class": {"linear": "LpBw",
                  "tabular": "TpBw"},
    },
    "mb_c_bwfw_MLE": {
        "task_type": "prediction",
        "control_agent": "q",
        "pg": False,
        "run_mode": "mb_c_bwfw_mle",
        "planning_iter": 1,
        "latent": False,
        "num_hidden_layers": 0,
        "num_units": 0,
        "model_family": "MLE",
        "target_networks": False,
        "class": {"linear": "LpBwFwMLE",
                  "tabular": "TpBwFwMLE"},
    },
    "mb_p_bw": {
        "task_type": "prediction",
        "control_agent": "q",
        "pg": False,
        "run_mode": "mb_p_bw",
        "planning_iter": 1,
        "latent": False,
        "num_hidden_layers": 0,
        "num_units": 0,
        "model_family": "MLE",
        "target_networks": False,
        "class": {"linear": "LpBw",
                  "tabular": "TpBw"},
    },
    "mb_c_true_bw": {
        "task_type": "prediction",
        "control_agent": "q",
        "pg": False,
        "run_mode": "mb_c_true_bw",
        "planning_iter": 1,
        "latent": False,
        "num_hidden_layers": 0,
        "num_units": 0,
        "model_family": "true",
        "target_networks": False,
        "class": {"linear": "LpTrueBw",
                  "tabular": "TpTrueBw"},
    },
    "mb_c_true_bwfw": {
        "task_type": "prediction",
        "control_agent": "q",
        "pg": False,
        "run_mode": "mb_c_true_bwfw",
        "planning_iter": 1,
        "latent": False,
        "num_hidden_layers": 0,
        "num_units": 0,
        "model_family": "true",
        "target_networks": False,
        "class": {"linear": "LpTrueBwFw",
                  "tabular": "TpTrueBwFw"},
    },
    "mb_p_true_bw": {
        "task_type": "prediction",
        "control_agent": "q",
        "pg": False,
        "run_mode": "mb_p_true_bw",
        "planning_iter": 1,
        "latent": False,
        "num_hidden_layers": 0,
        "num_units": 0,
        "model_family": "true",
        "target_networks": False,
        "class": {"linear": "LpTrueBw",
                  "tabular": "TpTrueBw"},
    },

    #### FW ####
    "p_fw": {
        "task_type": "prediction",
        "control_agent": "q",
        "pg": False,
        "run_mode": "p_fw",
        "planning_iter": 1,
        "latent": False,
        "num_hidden_layers": 0,
        "num_units": 0,
        "model_family": "MLE",
        "target_networks": False,
        "class": {"linear": "LpFw",
                  "tabular": "TpFw"},
    },
    "p_true_fw": {
        "task_type": "prediction",
        "control_agent": "q",
        "pg": False,
        "run_mode": "p_true_fw",
        "planning_iter": 1,
        "latent": False,
        "num_hidden_layers": 0,
        "num_units": 0,
        "model_family": "true",
        "target_networks": False,
        "class": {"linear": "LpTrueFw",
                  "tabular": "TpTrueFw"},
    },
    "c_fw": {
        "task_type": "prediction",
        "control_agent": "q",
        "pg": False,
        "run_mode": "c_fw",
        "planning_iter": 1,
        "latent": False,
        "num_hidden_layers": 0,
        "num_units": 0,
        "model_family": "MLE",
        "target_networks": False,
        "class": {"linear": "LpFw",
                  "tabular": "TpFw"},
    },
    "c_true_fw": {
        "task_type": "prediction",
        "control_agent": "q",
        "pg": False,
        "run_mode": "c_true_fw",
        "planning_iter": 1,
        "latent": False,
        "num_hidden_layers": 0,
        "num_units": 0,
        "model_family": "true",
        "target_networks": False,
        "class": {"linear": "LpTrueFw",
                  "tabular": "TpTrueFw"},
    },
    ### FW MLE ####
    "p_fw_MLE": {
        "task_type": "prediction",
        "control_agent": "q",
        "pg": False,
        "run_mode": "p_fw_mle",
        "planning_iter": 1,
        "latent": False,
        "num_hidden_layers": 0,
        "num_units": 0,
        "model_family": "MLE",
        "target_networks": False,
        "class": {"linear": "LpBw",
                  "tabular": "TpFwMLE"},
    },
    "c_fw_MLE": {
        "task_type": "prediction",
        "control_agent": "q",
        "pg": False,
        "run_mode": "c_fw_mle",
        "planning_iter": 1,
        "latent": False,
        "num_hidden_layers": 0,
        "num_units": 0,
        "model_family": "MLE",
        "target_networks": False,
        "class": {"linear": "LpBw",
                  "tabular": "TpFwMLE"},
    },

    ### FW MB ###
    "mb_c_fw": {
        "task_type": "prediction",
        "control_agent": "q",
        "pg": False,
        "run_mode": "mb_c_fw",
        "planning_iter": 1,
        "latent": False,
        "num_hidden_layers": 0,
        "num_units": 0,
        "model_family": "MLE",
        "target_networks": False,
        "class": {"linear": "LpFw",
                  "tabular": "TpFw"},
    },
    "mb_p_fw": {
        "task_type": "prediction",
        "control_agent": "q",
        "pg": False,
        "run_mode": "mb_p_fw",
        "planning_iter": 1,
        "latent": False,
        "num_hidden_layers": 0,
        "num_units": 0,
        "model_family": "MLE",
        "target_networks": False,
        "class": {"linear": "LpFw",
                  "tabular": "TpFw"},
    },
    "mb_c_fw_MLE": {
        "task_type": "prediction",
        "control_agent": "q",
        "pg": False,
        "run_mode": "mb_c_fw_mle",
        "planning_iter": 1,
        "latent": False,
        "num_hidden_layers": 0,
        "num_units": 0,
        "model_family": "MLE",
        "target_networks": False,
        "class": {"linear": "LpBw",
                  "tabular": "TpFwMLE"},
    },
    "mb_p_fw_MLE": {
        "task_type": "prediction",
        "control_agent": "q",
        "pg": False,
        "run_mode": "mb_p_fw_mle",
        "planning_iter": 1,
        "latent": False,
        "num_hidden_layers": 0,
        "num_units": 0,
        "model_family": "MLE",
        "target_networks": False,
        "class": {"linear": "LpBw",
                  "tabular": "TpFwMLE"},
    },
    "mb_c_true_fw": {
        "task_type": "prediction",
        "control_agent": "q",
        "pg": False,
        "run_mode": "mb_c_true_fw",
        "planning_iter": 1,
        "latent": False,
        "num_hidden_layers": 0,
        "num_units": 0,
        "model_family": "true",
        "target_networks": False,
        "class": {"linear": "LpTrueFw",
                  "tabular": "TpTrueFw"},
    },
    "mb_p_true_fw": {
        "task_type": "prediction",
        "control_agent": "q",
        "pg": False,
        "run_mode": "mb_p_true_fw",
        "planning_iter": 1,
        "latent": False,
        "num_hidden_layers": 0,
        "num_units": 0,
        "model_family": "true",
        "target_networks": False,
        "class": {"linear": "LpTrueFw",
                  "tabular": "TpTrueFw"},
    },

    ### BW FW ###
    "bw_fw": {
        "task_type": "prediction",
        "control_agent": "q",
        "pg": False,
        "run_mode": "fw_bw",
        "planning_iter": 1,
        "latent": False,
        "num_hidden_layers": 0,
        "num_units": 0,
        "model_family": "MLE",
        "target_networks": False,
        "class": {"linear": "LpBwFwExp",
                  "tabular": "TpBwFwDistrib"},
    },

    #### BW PAML ###
    "p_bw_PAML": {
        "task_type": "prediction",
        "control_agent": "q",
        "pg": False,
        "run_mode": "p_bw_PAML",
        "planning_iter": 1,
        "latent": False,
        "num_hidden_layers": 0,
        "num_units": 0,
        "model_family": "PAML",
        "target_networks": False,
        "class": {"linear": "LpBwPAML",
                  "tabular": "TpBwPAML"},
    },
    "p_bw_proj_PAML": {
        "task_type": "prediction",
        "control_agent": "q",
        "pg": False,
        "run_mode": "p_bw_proj_PAML",
        "planning_iter": 1,
        "latent": False,
        "num_hidden_layers": 0,
        "num_units": 0,
        "model_family": "PAML",
        "target_networks": False,
        "class": {"linear": "LpBwProjPAML",
                  "tabular": "TpBwProjPAML"},
    },
    "c_bw_PAML": {
        "task_type": "prediction",
        "control_agent": "q",
        "pg": False,
        "run_mode": "c_bw_PAML",
        "planning_iter": 1,
        "latent": False,
        "num_hidden_layers": 0,
        "num_units": 0,
        "model_family": "PAML",
        # "target_networks": True,
        "target_networks": False,
        "class": {"linear": "LpBwPAML",
                  "tabular": "TpBwPAML"},
    },
    "mb_c_bw_PAML": {
        "task_type": "prediction",
        "control_agent": "q",
        "pg": False,
        "run_mode": "mb_c_bw_PAML",
        "planning_iter": 1,
        "latent": False,
        "num_hidden_layers": 0,
        "num_units": 0,
        "model_family": "PAML",
        # "target_networks": True,
        "target_networks": False,
        "class": {
            "tabular":"TpBwPAML",
            "linear": "LpBwPAML"},
    },
    "mb_p_bw_PAML": {
        "task_type": "prediction",
        "control_agent": "q",
        "pg": False,
        "run_mode": "mb_p_bw_PAML",
        "planning_iter": 1,
        "latent": False,
        "num_hidden_layers": 0,
        "num_units": 0,
        "model_family": "PAML",
        # "target_networks": True,
        "target_networks": False,
        "class": {"linear": "LpBwPAML"},
    },

    #### FW PAML
    "p_fw_PAML": {
        "task_type": "prediction",
        "control_agent": "q",
        "pg": False,
        "run_mode": "p_fw_PAML",
        "planning_iter": 1,
        "latent": False,
        "num_hidden_layers": 0,
        "num_units": 0,
        "model_family": "PAML",
        # "target_networks": True,
        "target_networks": False,
        "class": {"tabular": "TpFwPAML",
                   "linear": "LpFwPAML"},
    },
    "c_fw_PAML": {
        "task_type": "prediction",
        "control_agent": "q",
        "pg": False,
        "run_mode": "c_fw_PAML",
        "planning_iter": 1,
        "latent": False,
        "num_hidden_layers": 0,
        "num_units": 0,
        "model_family": "PAML",
        # "target_networks": True,
        "target_networks": False,
        "class": {"tabular": "TpFwPAML",
                  "linear": "LpFwPAML"},
    },
    "mb_c_fw_PAML": {
        "task_type": "prediction",
        "control_agent": "q",
        "pg": False,
        "run_mode": "mb_c_fw_PAML",
        "planning_iter": 1,
        "latent": False,
        "num_hidden_layers": 0,
        "num_units": 0,
        "model_family": "PAML",
        # "target_networks": True,
        "target_networks": False,
        "class": {"linear": "LpFwPAML"},
    },
    "mb_p_fw_PAML": {
        "task_type": "prediction",
        "control_agent": "q",
        "pg": False,
        "run_mode": "mb_p_fw_PAML",
        "planning_iter": 1,
        "latent": False,
        "num_hidden_layers": 0,
        "num_units": 0,
        "model_family": "PAML",
        # "target_networks": True,
        "target_networks": False,
        "class": {"linear": "LpFwPAML"},
    },

}

##### STUFF WHICH I've used at some point, and works but not anymore ###
# "bw_iter": {
    #     "task_type": "prediction",
    #     "control_agent": "q",
    #     "pg": False,
    #     "run_mode": "bw_iter",
    #     "latent": False,
    #     "num_hidden_layers": 0,
    #     "num_units": 0,
    #     "model_family": "MLE",
    #     "target_networks": False,
    #     "class": {"linear": "LpExplicitIterat",
    #               "tabular": "TpExplicitIterat"},
    # },
    # "bw_fw_PWMA": {
    #     "task_type": "prediction",
    #     "control_agent": "q",
    #     "pg": False,
    #     "run_mode": "bw_fw_PWMA",
    #     "planning_iter": 1,
    #     "latent": False,
    #     "num_hidden_layers": 0,
    #     "num_units": 0,
    #     "model_family": "MLE",
    #     "target_networks": False,
    #     "class": {"linear": "LpFwBwPWMA",
    #               "tabular": "TpFwBwPWMA"},
    # },
    # "bw_fw_MG": {
    #     "task_type": "prediction",
    #     "control_agent": "q",
    #     "pg": False,
    #     "run_mode": "bw_fw_MG",
    #     "planning_iter": 1,
    #     "latent": False,
    #     "num_hidden_layers": 0,
    #     "num_units": 0,
    #     "model_family": "MLE",
    #     "target_networks": False,
    #     "class": {"linear": "LpFwBwMG",
    #               "tabular": "TpFwBwMG"},
    # },
    # "fw_rnd": {
    #     "task_type": "prediction",
    #     "control_agent": "q",
    #     "pg": False,
    #     "run_mode": "fw_rnd",
    #     "planning_iter": 1,
    #     "num_hidden_layers": 0,
    #     "num_units": 0,
    #     "model_family": "MLE",
    #     "latent": False,
    #     "target_networks": False,
    #     "class": {"linear": "LpFwRnd",
    #               "tabular": "TpFwRnd"},
    # },
    # "fw_pri": {
    #     "task_type": "prediction",
    #     "control_agent": "q",
    #     "pg": False,
    #     "run_mode": "fw_pri",
    #     "planning_iter": 1,
    #     "num_hidden_layers": 0,
    #     "num_units": 0,
    #     "model_family": "MLE",
    #     "latent": False,
    #     "target_networks": False,
    #     "class": {"linear": "LpFwPri",
    #               "tabular": "TpFwPri"},
    # },
# "q": {
#         "task_type": "control",
#         "pg": False,
#         "run_mode": "q",
#         "planning_depth": 0,
#         "planning_iter": 1,
#         "latent": False,
#         "num_hidden_layers": 0,
#         "num_units": 0,
#         "model_family": "q",
#         "target_networks": False,
#         "class": {"linear": "VanillaQ",
#                   "tabular": ""}
#     },
# "true_mult_bw": {
#     "task_type": "prediction",
#     "control_agent": "q",
#     "pg": False,
#     "run_mode": "true_mult_bw",
#     "planning_iter": 1,
#     "latent": False,
#     "num_hidden_layers": 0,
#     "num_units": 0,
#     "model_family": "true_mult",
#     "target_networks": False,
#     "class": {"linear": "LpTrueMultBw",
#               "tabular": "TpBw"},
# },
# "mult_fw": {
#     "task_type": "prediction",
#     "control_agent": "q",
#     "pg": False,
#     "run_mode": "fw_mult",
#     "planning_iter": 1,
#     "latent": False,
#     "num_hidden_layers": 0,
#     "num_units": 0,
#     "model_family": "mult_MLE",
#     "target_networks": False,
#     "class": {"linear": "LpMultFw",
#               "tabular": "TpFwMult"},
# },
# "latent_bw_PAML": {
#         "task_type": "prediction",
#         "control_agent": "q",
#         "pg": False,
#         "run_mode": "latent_bw_PAML",
#         "planning_iter": 1,
#         "latent": True,
#         "num_hidden_layers": 0,
#         "num_units": 32,
#         "model_family": "PAML",
#         # "target_networks": True,
#         "target_networks": False,
#         "class": {"linear": "LpBwPAML"},
#     },
# "bw_update": {
#     "task_type": "prediction",
#     "control_agent": "q",
#     "pg": False,
#     "run_mode": "bw_update",
#     "planning_iter": 1,
#     "latent": False,
#     "num_hidden_layers": 0,
#     "num_units": 0,
#     "model_family": "update",
#     # "target_networks": True,
#     "target_networks": False,
#     "class": {"linear": "LpBwUpdate"},
# },
# "bw_mult_PAML": {
#     "task_type": "prediction",
#     "control_agent": "q",
#     "pg": False,
#     "run_mode": "bw_mult_PAML",
#     "planning_iter": 1,
#     "latent": False,
#     "num_hidden_layers": 0,
#     "num_units": 0,
#     "model_family": "mult_PAML",
#     # "target_networks": True,
#     "target_networks": False,
#     "class": {"linear": "LpBwMultPAML"},
# },
# "bw_vaware": {
#     "task_type": "prediction",
#     "control_agent": "q",
#     "pg": False,
#     "run_mode": "bw_vaware",
#     "planning_iter": 1,
#     "latent": False,
#     "num_hidden_layers": 0,
#     "num_units": 0,
#     "model_family": "PAML",
#     # "target_networks": True,
#     "target_networks": False,
#     "class": {"linear": "LpBWValueAware"},
# },
# "bw_meta": {
#     "task_type": "prediction",
#     "control_agent": "q",
#     "pg": False,
#     "run_mode": "bw_meta",
#     "planning_iter": 1,
#     "latent": False,
#     "num_hidden_layers": 0,
#     "num_units": 0,
#     "model_family": "PAML",
#     # "target_networks": True,
#     "target_networks": False,
#     "class": {"linear": "LpBwMeta"},
# },
# "ac_vanilla": {
#     "task_type": "control",
#     "pg": True,
#     "run_mode": "ac_vanilla",
#     "planning_depth": 0,
#     "planning_iter": 1,
#     "model_family": "ac",
#     "latent": False,
#     "num_hidden_layers": 0,
#     "num_units": 0,
#     "target_networks": False,
#     "class": {"linear": "ACVanilla"}
# },
# "pg": {
#     "task_type": "control",
#     "pg": True,
#     "run_mode": "pg",
#     "planning_depth": 0,
#     "planning_iter": 1,
#     "model_family": "ac",
#     "latent": False,
#     "num_hidden_layers": 0,
#     "num_units": 0,
#     "target_networks": False,
#     "class": {"linear": "PG"}
# },
# "fw_mult_PAML": {
#         "task_type": "prediction",
#         "control_agent": "q",
#         "pg": False,
#         "run_mode": "fw_mult_PAML",
#         "planning_iter": 1,
#         "latent": False,
#         "num_hidden_layers": 0,
#         "num_units": 0,
#         "model_family": "mult_PAML",
#         # "target_networks": True,
#         "target_networks": False,
#         "class": {"linear": "LpFwMultPAML"},
#     },
# "mult_bw": {
#         "task_type": "prediction",
#         "control_agent": "q",
#         "pg": False,
#         "run_mode": "mult_bw",
#         "planning_iter": 1,
#         "latent": False,
#         "num_hidden_layers": 0,
#         "num_units": 0,
#         "model_family": "mult_MLE",
#         "target_networks": False,
#         "class": {"linear": "LpMultBw",
#                   "tabular": "TpBw"},
#     },