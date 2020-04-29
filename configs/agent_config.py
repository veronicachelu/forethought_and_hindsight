config = {
    "mult_bw": {
        "task_type": "prediction",
        "control_agent": "q",
        "pg": False,
        "run_mode": "mult_bw",
        "planning_iter": 1,
        "latent": False,
        "num_hidden_layers": 0,
        "num_units": 0,
        "model_family": "mult_extrinsic",
        "target_networks": False,
        "class": {"linear": "LpMultBw",
                  "tabular": "TpExplicitDistrib"},
    },
    "true_bw": {
        "task_type": "prediction",
        "control_agent": "q",
        "pg": False,
        "run_mode": "true_bw",
        "planning_iter": 1,
        "latent": False,
        "num_hidden_layers": 0,
        "num_units": 0,
        "model_family": "true",
        "target_networks": False,
        "class": {"linear": "LpTrueBw",
                  "tabular": "TpExplicitDistrib"},
    },
    "true_mult_bw": {
        "task_type": "prediction",
        "control_agent": "q",
        "pg": False,
        "run_mode": "true_mult_bw",
        "planning_iter": 1,
        "latent": False,
        "num_hidden_layers": 0,
        "num_units": 0,
        "model_family": "true_mult",
        "target_networks": False,
        "class": {"linear": "LpTrueMultBw",
                  "tabular": "TpExplicitDistrib"},
    },
    "q": {
        "task_type": "control",
        "pg": False,
        "run_mode": "q",
        "planning_depth": 0,
        "planning_iter": 1,
        "latent": False,
        "num_hidden_layers": 0,
        "num_units": 0,
        "model_family": "q",
        "target_networks": False,
        "class": {"linear": "VanillaQ",
                  "tabular": ""}
    },
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
        "model_family": "extrinsic",
        "target_networks": False,
        "class": {"linear": "LpVanilla",
                 "tabular": "TpVanilla"}
        },
    "vanilla_intr": {
        "task_type": "prediction",
        "control_agent": "q",
        "pg": False,
        "run_mode": "vanilla_intr",
        "planning_depth": 0,
        "planning_iter": 1,
        "model_family": "intrinsic",
        "latent": False,
        "num_hidden_layers": 0,
        "num_units": 0,
        "target_networks": False,
        "class": {"linear": "LpIntrinsicVanilla"}
    },
    "bw": {
        "task_type": "prediction",
        "control_agent": "q",
        "pg": False,
        "run_mode": "bw_extr",
        "planning_iter": 1,
        "latent": False,
        "num_hidden_layers": 0,
        "num_units": 0,
        "model_family": "extrinsic",
        "target_networks": False,
        "class": {"linear": "LpBw",
                  "tabular": "TpExplicitDistrib"},
    },
    "fw": {
        "task_type": "prediction",
        "control_agent": "q",
        "pg": False,
        "run_mode": "fw",
        "planning_iter": 1,
        "latent": False,
        "num_hidden_layers": 0,
        "num_units": 0,
        "model_family": "extrinsic",
        "target_networks": False,
        "class": {"linear": "LpFw",
                  "tabular": "TpFw"},
    },
    "bw_fw": {
        "task_type": "prediction",
        "control_agent": "q",
        "pg": False,
        "run_mode": "fw_bw",
        "planning_iter": 1,
        "latent": False,
        "num_hidden_layers": 0,
        "num_units": 0,
        "model_family": "extrinsic",
        "target_networks": False,
        "class": {"linear": "LpBwFwExp",
                  "tabular": "TpBwFwDistrib"},
    },
    "bw_iter": {
        "task_type": "prediction",
        "control_agent": "q",
        "pg": False,
        "run_mode": "bw_iter",
        "latent": False,
        "num_hidden_layers": 0,
        "num_units": 0,
        "model_family": "extrinsic",
        "target_networks": False,
        "class": {"linear": "LpExplicitIterat",
                  "tabular": "TpExplicitIterat"},
    },
    "bw_fw_PWMA": {
        "task_type": "prediction",
        "control_agent": "q",
        "pg": False,
        "run_mode": "bw_fw_PWMA",
        "planning_iter": 1,
        "latent": False,
        "num_hidden_layers": 0,
        "num_units": 0,
        "model_family": "extrinsic",
        "target_networks": False,
        "class": {"linear": "LpFwBwPWMA",
                  "tabular": "TpFwBwPWMA"},
    },
    "bw_fw_MG": {
        "task_type": "prediction",
        "control_agent": "q",
        "pg": False,
        "run_mode": "bw_fw_MG",
        "planning_iter": 1,
        "latent": False,
        "num_hidden_layers": 0,
        "num_units": 0,
        "model_family": "extrinsic",
        "target_networks": False,
        "class": {"linear": "LpFwBwMG",
                  "tabular": "TpFwBwMG"},
    },
    "fw_rnd": {
        "task_type": "prediction",
        "control_agent": "q",
        "pg": False,
        "run_mode": "fw_rnd",
        "planning_iter": 1,
        "num_hidden_layers": 0,
        "num_units": 0,
        "model_family": "extrinsic",
        "latent": False,
        "target_networks": False,
        "class": {"linear": "LpFwRnd",
                  "tabular": "TpFwRnd"},
    },
    "fw_pri": {
        "task_type": "prediction",
        "control_agent": "q",
        "pg": False,
        "run_mode": "fw_pri",
        "planning_iter": 1,
        "num_hidden_layers": 0,
        "num_units": 0,
        "model_family": "extrinsic",
        "latent": False,
        "target_networks": False,
        "class": {"linear": "LpFwPri",
                  "tabular": "TpFwPri"},
    },

    "ac_vanilla": {
        "task_type": "control",
        "pg": True,
        "run_mode": "ac_vanilla",
        "planning_depth": 0,
        "planning_iter": 1,
        "model_family": "ac",
        "latent": False,
        "num_hidden_layers": 0,
        "num_units": 0,
        "target_networks": False,
        "class": {"linear": "ACVanilla"}
    },
    "pg": {
        "task_type": "control",
        "pg": True,
        "run_mode": "pg",
        "planning_depth": 0,
        "planning_iter": 1,
        "model_family": "ac",
        "latent": False,
        "num_hidden_layers": 0,
        "num_units": 0,
        "target_networks": False,
        "class": {"linear": "PG"}
    },
    "latent_bw_intr": {
        "task_type": "prediction",
        "control_agent": "q",
        "pg": False,
        "run_mode": "latent_bw_intr",
        "planning_iter": 1,
        "latent": True,
        "num_hidden_layers": 0,
        "num_units": 32,
        "model_family": "intrinsic",
        # "target_networks": True,
        "target_networks": False,
        "class": {"linear": "LpBwCorrBased"},
    },
    "bw_intr": {
        "task_type": "prediction",
        "control_agent": "q",
        "pg": False,
        "run_mode": "bw_intr",
        "planning_iter": 1,
        "latent": False,
        "num_hidden_layers": 0,
        "num_units": 0,
        "model_family": "intrinsic",
        # "target_networks": True,
        "target_networks": False,
        "class": {"linear": "LpBwCorrBased"},
    },
    "bw_vaware": {
        "task_type": "prediction",
        "control_agent": "q",
        "pg": False,
        "run_mode": "bw_vaware",
        "planning_iter": 1,
        "latent": False,
        "num_hidden_layers": 0,
        "num_units": 0,
        "model_family": "intrinsic",
        # "target_networks": True,
        "target_networks": False,
        "class": {"linear": "LpBWValueAware"},
    },
    "bw_meta": {
        "task_type": "prediction",
        "control_agent": "q",
        "pg": False,
        "run_mode": "bw_meta",
        "planning_iter": 1,
        "latent": False,
        "num_hidden_layers": 0,
        "num_units": 0,
        "model_family": "intrinsic",
        # "target_networks": True,
        "target_networks": False,
        "class": {"linear": "LpBWMeta"},
    },
    "random": {
        "task_type": "prediction",
        "control_agent": "q",
        "pg": False,
        "run_mode": "bw_intr",
        "planning_iter": 1,
        "latent": False,
        "num_hidden_layers": 0,
        "num_units": 0,
        "model_family": "intrinsic",
        # "target_networks": True,
        "target_networks": False,
        "class": {"linear": "LpExplicitValueBased"},
    },
    "learned": {
        "task_type": "prediction",
        "control_agent": "q",
        "pg": False,
        "run_mode": "bw_intr",
        "planning_iter": 1,
        "latent": False,
        "num_hidden_layers": 0,
        "num_units": 0,
        "model_family": "intrinsic",
        # "target_networks": True,
        "target_networks": False,
        "class": {"linear": "LpExplicitValueBased"},
    },
    "fw_intr": {
        "task_type": "prediction",
        "control_agent": "q",
        "pg": False,
        "run_mode": "fw_intr",
        "planning_iter": 1,
        "latent": False,
        "num_hidden_layers": 0,
        "num_units": 0,
        "model_family": "intrinsic",
        # "target_networks": True,
        "target_networks": False,
        "class": {"linear": "LpFwValueBased"},
    },
}