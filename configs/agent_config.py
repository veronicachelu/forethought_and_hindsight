config = {
    "vanilla": {
        "run_mode": "vanilla",
        "planning_depth": 0,
        "planning_iter": 1,
        "class": {"linear": "LpVanilla",
                 "tabular": "TpVanilla"}
        },
    "bw": {
        "run_mode": "explicit_exp",
        "planning_iter": 1,
        "class": {"linear": "LpExplicitExp",
                  "tabular": "TpExplicitDistrib"},
    },
    "bw_fw": {
        "run_mode": "fw_bw_exp",
        "planning_iter": 1,
        "class": {"linear": "LpBwFwExp",
                  "tabular": "TpBwFwDistrib"},
    },
    "bw_iter": {
        "run_mode": "bw_iter",
        "class": {"linear": "LpExplicitIterat",
                  "tabular": "TpExplicitIterat"},
    },
    "bw_fw_PWMA": {
        "run_mode": "bw_fw_PWMA",
        "planning_iter": 1,
        "class": {"linear": "LpFwBwPWMA",
                  "tabular": "TpFwBwPWMA"},
    },
    "bw_fw_MG": {
        "run_mode": "bw_fw_MG",
        "planning_iter": 1,
        "class": {"linear": "LpFwBwMG",
                  "tabular": "TpFwBwMG"},
    },
    "fw_rnd": {
        "run_mode": "fw_rnd",
        "planning_iter": 1,
        "planning_depth": 1,
        "lr_m": 0,
        "class": {"linear": "LpFwRnd",
                  "tabular": "TpFwRnd"},
    },
    "fw_pri": {
        "run_mode": "fw_pri",
        "planning_iter": 1,
        "planning_depth": 1,
        "lr_m": 0,
        "class": {"linear": "LpFwPri",
                  "tabular": "TpFwPri"},
    },
    
}