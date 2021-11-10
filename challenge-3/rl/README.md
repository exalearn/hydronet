# Reinforcement Learning for Graph Optimization

This folder contains codes for running experiments with reinforcement learning for optimizing graph structures.

The core piece is the `run_ppo.py` code which runs experiments with different settings, such as:

- `--max-size`: the target water cluster size for the optimizer
- `--driver-episodes`: number of episodes to perform with the driver
- `--ppo-entropy-regularizer`: penalty term for entropy for the PPO policy learner

Call `python run_ppo.py -h` for a full list of options.

## Analyzing outputs

`run_ppo` writes outputs for each run into a separate subdirectory of `runs`.

The outputs include:
- `collect_policy`: The graph sampling policy saved in SaveModel format
- `final_trajs`: A Pandas dataframe of a few trajectories sampled from the model
- `log.csv`: Logging data from trianing
- `run_params.json`: The settings used to start the run
