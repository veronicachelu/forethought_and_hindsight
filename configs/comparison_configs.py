configs={
    "repeat": {
        "non_parametric_fw": {
            "agents": ["fw_rnd", "fw_pri"],
            "planning_depths": [1, 1],
            "replay_buffers": [50, 50]
        },

        "non_parametric_fw_bw": {
            "agents": ["bw_fw_PWMA", "bw_fw_MG"],
            "planning_depths": [1, 1],
            "replay_buffers": [50, 50]
        },

        "parametric": {
            "agents": ["bw", "bw_fw"],
            "planning_depths": [1, 1],
            "replay_buffers": [0, 0],
        },
        
        "final": {
            "agents": ["bw", "fw_rnd", "bw_fw_PWMA"],
            "planning_depths": [1, 1, 1],
            "replay_buffers": [0, 50, 50]
        }
    }
}