from absl import app
from absl import flags
from jax import random as jrandom
from tqdm import tqdm
from main_utils import *
import agents
import experiment
import network
import utils
from utils import *


def run_agent(env, agent, logs, aux_agent_configs, ignore_existent=True):
    persistent_agent_config = configs.agent_config.config[agent["agent"]]
    agent_run_mode = "{}_{}_{}".format(persistent_agent_config["run_mode"], agent["planning_depth"],
                                       agent["replay_capacity"])
    agent_logs = os.path.join(logs, '{}/summaries/'.format(agent_run_mode))

    env_config, _ = load_env_and_volatile_configs(env)

    seed_config = {"planning_depth": int(agent["planning_depth"]),
                   "replay_capacity": int(agent["replay_capacity"]),
                   "lr": float(agent["lr"]),
                   "lr_m": float(agent["lr_m"]),
                   "lr_p": float(agent["lr_m"])}

    # for seed in tqdm(range(0, env_config["num_runs"])):
    for seed in range(0, env_config["num_runs"]):
        seed_config["seed"] = seed
        space = {
            "logs": logs,
            "plot_errors": False,
            "plot_values": False,
            "plot_curves": True,
            "log_period": aux_agent_configs["log_period"],
            "env_config": env_config,
            "agent_config": persistent_agent_config,
            "crt_config": seed_config}

        if not ignore_existent:
            agent_logs_seed = os.path.join(agent_logs, 'seed_{}'.format(seed))
            if os.path.exists(agent_logs_seed) and \
                            len(os.listdir(agent_logs_seed)) > 0:
                continue
        run_objective(space, aux_agent_configs)


def run_objective(space, aux_agent_configs):
    seed = space["crt_config"]["seed"]
    env, agent, mdp_solver = run_experiment(seed, space, aux_agent_configs)
    total_rmsve, final_rmsve, start_rmsve, avg_steps, values, errors = experiment.run_episodic(
        agent=agent,
        space=space,
        aux_agent_configs=aux_agent_configs,
        mdp_solver=mdp_solver,
        environment=env,
    )

