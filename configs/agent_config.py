config = {
    "vanilla_intr": {
        "run_mode": "vanilla_intr",
        "planning_depth": 0,
        "planning_iter": 1,
        "latent": False,
        "class": {"linear": "LpIntrinsicVanilla"}
    },
    "latent_vanilla_intr": {
        "run_mode": "latent_vanilla_intr",
        "planning_depth": 0,
        "planning_iter": 1,
        "latent": True,
        "class": {"linear": "LpIntrinsicVanilla"}
    },
    "vanilla": {
        "run_mode": "vanilla",
        "planning_depth": 0,
        "planning_iter": 1,
        "latent": False,
        "class": {"linear": "LpVanilla",
                 "tabular": "TpVanilla"}
        },
    "latent_bw_intr": {
        "run_mode": "latent_bw_intr",
        "planning_iter": 1,
        "latent": True,
        "class": {"linear": "LpExplicitValueBased"},
    },
    "bw_intr": {
        "run_mode": "bw_intr",
        "planning_iter": 1,
        "latent": False,
        "class": {"linear": "LpExplicitValueBased"},
    },
    "bw": {
        "run_mode": "bw_extr",
        "planning_iter": 1,
        "latent": False,
        "class": {"linear": "LpExplicitExp",
                  "tabular": "TpExplicitDistrib"},
    },
    "bw_fw": {
        "run_mode": "fw_bw",
        "planning_iter": 1,
        "latent": False,
        "class": {"linear": "LpBwFwExp",
                  "tabular": "TpBwFwDistrib"},
    },
    "bw_iter": {
        "run_mode": "bw_iter",
        "latent": False,
        "class": {"linear": "LpExplicitIterat",
                  "tabular": "TpExplicitIterat"},
    },
    "bw_fw_PWMA": {
        "run_mode": "bw_fw_PWMA",
        "planning_iter": 1,
        "latent": False,
        "class": {"linear": "LpFwBwPWMA",
                  "tabular": "TpFwBwPWMA"},
    },
    "bw_fw_MG": {
        "run_mode": "bw_fw_MG",
        "planning_iter": 1,
        "latent": False,
        "class": {"linear": "LpFwBwMG",
                  "tabular": "TpFwBwMG"},
    },
    "fw_rnd": {
        "run_mode": "fw_rnd",
        "planning_iter": 1,
        "planning_depth": 1,
        "lr_m": 0,
        "latent": False,
        "class": {"linear": "LpFwRnd",
                  "tabular": "TpFwRnd"},
    },
    "fw_pri": {
        "run_mode": "fw_pri",
        "planning_iter": 1,
        "planning_depth": 1,
        "lr_m": 0,
        "latent": False,
        "class": {"linear": "LpFwPri",
                  "tabular": "TpFwPri"},
    },
    
}