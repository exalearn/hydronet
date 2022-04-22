"""Generate graphs given the results of a certain RL training, invert them, upload result to database"""
from argparse import ArgumentParser
from pathlib import Path
import pickle as pkl
import os

import tensorflow as tf
import pandas as pd
from ase.optimize import BFGS
from tf_agents.agents import PPOClipAgent
from tf_agents.environments import TFPyEnvironment
from tqdm import tqdm

from hydronet.inversion import measure_adj_difference
from hydronet.inversion.force import convert_directed_graph_to_xyz
from hydronet.db import HydroNetDB, HydroNetRecord
from ttm.ase import TTMCalculator


def get_trajectories(environment, policy, num_episodes=10) -> pd.DataFrame:
    """Get trajectory of energy wrt step

    Args:
        environment: Water cluster environment
        policy: Policy to execute
    Returns:
        List of trajectories
    """
    output = []
    for e in tqdm(range(num_episodes), desc='Generating graphs'):
        time_step = environment.reset()
        step = 0
        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            graph = tf_env.pyenv.envs[0].get_state()
            output.append({
                'episode': e,
                'step': step,
                'graph': graph,
                'reward': time_step.reward.numpy(),
                'size': len(graph)
            })
            step += 1

    return pd.DataFrame(output)


if __name__ == '__main__':
    parser = ArgumentParser()

    # Basic arguments
    parser.add_argument('run_dir', help='Path to a directory with a trained RL policy', type=Path)
    parser.add_argument('episode_count', help='Number of episodes to run', type=int)

    # Define the options for how many to validate

    args = parser.parse_args()

    # Check that the run directory exists
    run_dir = args.run_dir
    assert run_dir.is_dir(), f'{run_dir} does not exist'

    # Load in the environment
    with open(run_dir / 'env.pkl', 'rb') as fp:
        env = pkl.load(fp)
    tf_env = TFPyEnvironment(env)

    # Load in the policies
    with open(run_dir / 'actor_network.pkl', 'rb') as fp:
        actor_net = pkl.load(fp)
    with open(run_dir / 'critic_network.pkl', 'rb') as fp:
        critic_net = pkl.load(fp)

    # Use it to create the PPO agent
    #  We use clipping to avoid problems with exploding KL gradients
    tf_agent = PPOClipAgent(
        tf_env.time_step_spec(),
        tf_env.action_spec(),
        actor_net=actor_net,
        value_net=critic_net,
        optimizer=tf.keras.optimizers.Adam(),
        normalize_observations=False,
    )
    tf_agent.initialize()

    # Run the desired number of episodes
    rl_traj = get_trajectories(tf_env, tf_agent.collect_policy, args.episode_count)

    # Run the inversions on them
    for graph in tqdm(rl_traj['graph']):
        initial_guesses = [
            convert_directed_graph_to_xyz(graph) for _ in range(1)
        ]

        # Relax them all with TTM
        _ttm = TTMCalculator()


        def _relax_geom(atoms):
            atoms = atoms.copy()
            atoms.set_calculator(_ttm)
            dyn = BFGS(atoms, logfile=os.devnull)
            dyn.run(fmax=0.05, steps=1024)
            return atoms

        relaxed = [
            _relax_geom(x) for x in initial_guesses
        ]

        # Measure whether we get the same graph
        records = [
            HydroNetRecord.from_atoms(x) for x in relaxed
        ]
        n_different = [
            measure_adj_difference(graph, record.coarse_nx) for record in records
        ]

        # Get the graph with the least number of mi
        if min(n_different) != 0:
            break
