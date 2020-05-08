#### (first readme got deleted for mysterious reasons, this is a second try)
### Requirements
python 3.6 (might work with other versions as well, but haven't tried)
tensorflow 2.0 (or 2.1 last version I guess)
jax
dm-env
others probablly but haven't kept task

### Usage

#### run

```` python run.py --agent=AGENT --env=ENV --lr=LR_DIRECT_LEARNING --lr_p=LR_PLANNING --lr_m=LR_MODEL --planning_depth=PLANNING_DEPTH````

* AGENT
    * for direct learning + planning, I follow this convention: AGENT = PIVOT+NAME_OF_AGENT
    * for pure model based, I follow AGENT=mb_PIVOT_NAME_OF_AGENT
    * pivot = p or c (previous or current) depending on which transition is going to be used for planning
    * NAME_OF_AGENT 
        * for tabular can be one of the following: vanilla, bw_MLE, fw_MLE, true_bw, true_fw, bw_recur_MLE
        * for linear can be one of the following: vanilla, bw, fw, true_bw, bw_PAML, fw_PAML
* PLANNING_DEPTH refers to the model horizon and should be 0 for vanilla and 1 for the rest (can be larger than 1 for n-step models)
* tabular: ENV=split, LR_DIRECT_LEARNING=LR_PLANNING=LR_MODEL=0.1
* linear: ENV=obstacle, LR_DIRECT_LEARNING=LR_PLANNING=0.001, LR_MODEL=0.005
* linear: ENV=reg1_obstacle, reg2_obstacle, reg01_obstacle are versions of obstacle where there are 8 additional irrelevant dimensions to the observation sapce and I add also L1, and L2 parameter regularization respecively to the model parameters to constrain the function class, reg01_obstacle uses a 0.1 stength for L1 reg, the reg_obstalce uses 1.0 for L1 reg and reg2_obstalce uses 1.0 for L2 reg
* agent configs are in configs/agent_config.py
    
#### plot

````python plot_all.py --env=ENV --pivoting=PIVOTING --lr=LR --tabular=TABULAR --mle=MLE --mb=MB````

* ENV: same as above
* use TABULAR=True only for tabular and set the LR=0.1, otherwise set TABULAR=False and don't use the LR at all param
* MLE=True only for tabular (the rest should use an L2 loss, MLE is reserved for cross entropy)
* MB=True only for pure model-based learning
* PIVOTING: has to follow the format set in config/comparison_configs.py, e.g. "bw_p_fw_p", this is appended to "_all" and than takes the agents set in comparison configs at that key to plot them
* you can also see plots in tensorboad live, but this plotting makes an average over seeds and plots them nicely

* There is also hyperparamer tuning, which is why the super added complexity in configs and stuff, but that is potentially less interesting to add here