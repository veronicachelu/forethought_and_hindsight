config = {
    "ac_vanilla": {
        "pg": True,
        "run_mode": "ac_vanilla",
        "planning_depth": 0,
        "planning_iter": 1,
        "model_family": "extrinsic",
        "latent": False,
        "num_hidden_layers": 0,
        "num_units": 80,
        "target_networks": False,
        "class": {"linear": "ACVanilla"}
    },
    "vanilla_intr": {
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
    "latent_vanilla_intr": {
        "pg": False,
        "run_mode": "latent_vanilla_intr",
        "planning_depth": 0,
        "planning_iter": 1,
        "latent": True,
        "num_hidden_layers": 0,
        "num_units": 80,
        "model_family": "intrinsic",
        "target_networks": False,
        "class": {"linear": "LpIntrinsicVanilla"}
    },
    "vanilla": {
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
    "latent_bw_intr": {
        "pg": False,
        "run_mode": "latent_bw_intr",
        "planning_iter": 1,
        "latent": True,
        "num_hidden_layers": 0,
        "num_units": 80,
        "model_family": "intrinsic",
        "target_networks": True,
        "class": {"linear": "LpExplicitValueBased"},
    },
    "bw_intr": {
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
    "bw": {
        "pg": False,
        "run_mode": "bw_extr",
        "planning_iter": 1,
        "latent": False,
        "num_hidden_layers": 0,
        "num_units": 0,
        "model_family": "extrinsic",
        "target_networks": False,
        "class": {"linear": "LpExplicitExp",
                  "tabular": "TpExplicitDistrib"},
    },
    "fw": {
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
    
}