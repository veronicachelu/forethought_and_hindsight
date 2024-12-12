### Requirements
python 3.6
tensorflow 2.0
jax
dm-env

### Usage

#### Fan-in/Fan-out experiment

```` python run_tabular.py --agent=AGENT --env=ENV --lr=LR_DIRECT_LEARNING --lr_p=LR_PLANNING --lr_m=LR_MODEL --planning_depth=PLANNING_DEPTH````

* AGENT
    * for direct learning + planning, we follow this convention: AGENT = REF-STATE+NAME-OF-AGENT
    * for pure model based, we follow AGENT=mb_REF-STATE_NAME-OF-AGENT
    * REF-STATE = p or c (previous=x or current=x_prime) depending on which end of the transition is going to be used for planning
    * NAME-OF-AGENT 
        * for tabular can be one of the following: vanilla, bw_MLE, fw_MLE, true_bw, true_fw
* PLANNING_DEPTH refers to the model horizon and should be 0 for vanilla and 1 for the rest (can be larger than 1 for n-step models)
* LR_DIRECT_LEARNING=LR_PLANNING=LR_MODEL=1.0

##### For the Right plot
* ENV can be one of the following: bipartitie_100_1, bipartite_10_1, bipartite, bipartite_1_10, bipartite_100 (corresponding to fan_in/fan_out: 100/1, 10/1, 1/1, 1/10, 1/100)
* E.g. 
```` python run_tabular.py --agent=bw_MLE --env=bipartitie_10_1 --lr=1.0 --lr_p=1.0 --lr_m=1.0 --planning_depth=1````

##### For the Left plot 
* ENV can be one of the following: bipartitie_100_10_1_2L, bipartite_1_10_100_2L (corresponding to fan_in/fan_out: 100/10/1, 1/10/100)
* E.g. 
```` python run_tabular.py --agent=bw_MLE --env=bipartitie_100_10_1_2L --lr=1.0 --lr_p=1.0 --lr_m=1.0 --planning_depth=1````

    
#### plot

#### for the Left plot
````python plot_custom_prediction.py````

### for the Right plot
````python plot_custom_auc.py````

#### Control Experiments

```` python q_control.py --agent=AGENT --env=ENV --lr_ctrl=LR_DIRECT_LEARNING --lr_m=LR_MODEL --planning_depth=PLANNING_DEPTH````
* AGENT
    * for direct learning + planning, we follow this convention: AGENT = REF-STATE+NAME-OF-AGENT_q
    * for pure model based, we follow AGENT=mb_REF-STATE_NAME-OF-AGENT_q
    * REF-STATE = p or c (previous=x or current=x_prime) depending on which end of the transition is going to be used for planning
    * NAME-OF-AGENT 
        * for tabular can be one of the following: vanilla, bw, fw, true_bw, true_fw
* PLANNING_DEPTH refers to the model horizon and should be 0 for vanilla and 1 for the rest (can be larger than 1 for n-step models)
* LR_CTRL, LR_MODEL as specified in the appendix of the paper

#### For the REF-STATE comparison (e.g. for the right plot)
```` python q_control.py --agent=p_fw_q --env=maze_1 --lr_ctrl=1.0 --lr_m=1.0 --planning_depth=1````
```` python q_control.py --agent=p_bw_q --env=maze_1 --lr_ctrl=1.0 --lr_m=1.0 --planning_depth=1````
```` python q_control.py --agent=c_fw_q --env=maze_1 --lr_ctrl=1.0 --lr_m=1.0 --planning_depth=1````
```` python q_control.py --agent=c_bw_q --env=maze_1 --lr_ctrl=1.0 --lr_m=1.0 --planning_depth=1````
* These are run on env maze_1 (deterministic)

#### For the control experiments in the stochastic environments
* Change the ENV from maze_1 (deterministic), maze_stoch (stochastic transitions), maze_05 (stochastic rewards with prob 0.5), maze_01 (stochastic rewards with prob 0.1)

### plots

#### For reference states experiments
````python plot_cum_reward_per_ref.py````

#### For control experiments
````python plot_custom_control.py````




