from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
import hashlib
import json
import csv

from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.drivers.dynamic_episode_driver import DynamicEpisodeDriver
from tf_agents.policies.policy_saver import PolicySaver
from tf_agents.agents import PPOClipAgent
from tqdm import tqdm
import tensorflow as tf
import pandas as pd

from hydronet.rl.tf.networks import GCPNActorNetwork, GCPNCriticNetwork
from hydronet.rl.tf.env import SimpleEnvironment
from hydronet.rl.rewards.mpnn import MPNNReward
from hydronet.mpnn.layers import custom_objects
from hydronet.utils import get_platform_info



# Get the default path for the MPNN
_mpnn_path = Path() / '..' / '..' / 'challenge-2' / 'best-model' / 'best_model.h5'


def make_reward(args):
    """Configure the reward function
    
    Args:
        args: Parsed user arguments
    Returns:
        - Reward function
        - Whether to return reward only for last step
    """
    if args.reward == 'mpnn_last':
        model = tf.keras.models.load_model(args.mpnn_path, custom_objects=custom_objects)
        reward = MPNNReward(model, per_water=False)
        return reward, True
    else:
        raise ValueError(f'Undefined reward function: {args.reward}')
        

def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):
        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


def get_trajectories(environment, policy, eng_reward, num_episodes=10) -> pd.DataFrame:
    """Get trajectory of energy wrt step
    
    Args:
        environment: Water cluster environment
        policy: Policy to execute
    Returns:
        List of trajectories
    """
    output = []
    for e in tqdm(range(num_episodes), desc='Post-training'):
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
                'energy': -eng_reward(graph),
                'size': len(graph),
                'reward': float(-1 * time_step.reward.numpy()),
            })
            step += 1

    return pd.DataFrame(output)


if __name__ == "__main__":
    # Make an options parser
    arg_parser = ArgumentParser()
    
    #   Group 1: Things related to the environment
    group = arg_parser.add_argument_group('Environment Options', 'Options related to the water cluster environment, such as reward structure')
    group.add_argument('--reward', choices=['mpnn_last'], default='mpnn_last', help='Name of the reward function to use. Rewards are defined in code')
    group.add_argument('--mpnn-path', default=str(_mpnn_path), help='Path to the MPNN used for evaluating energy, if needed')
    group.add_argument('--max-size', default=10, help='Maximum size of the water cluster.', type=int)
    
    #   Group 2: Related to the GCPN networks
    group = arg_parser.add_argument_group('GCPN Options', 'Options related to the GCPN networks')
    group.add_argument('--actor-messages', type=int, default=8, help='Number of message passing layers in actor network')
    group.add_argument('--actor-features', type=int, default=64, help='Number of features per node in actor network')
    group.add_argument('--actor-graph-features', action='store_true', help='Whether to include graph-level features when'
                                                                           ' generating action probabilities in actor network')
    group.add_argument('--critic-messages', type=int, default=8, help='Number of message passing layers in critic network')
    group.add_argument('--critic-features', type=int, default=64, help='Number of features per node in critic network')
    
    #   Group 3: Related to the PPO learner
    group = arg_parser.add_argument_group('PPO Options', 'Options related to the PPO learner')
    group.add_argument('--ppo-learning-rate', default=1e-3, type=float, help='Learning reate for PPO')
    group.add_argument('--ppo-entropy-regularizer', default=1e-4, type=float, help='Entropy regularization for PPO')
    
    #    Group 4: Related to the driver/learner
    group = arg_parser.add_argument_group('Driver Options', 'Options for the driver/learner')
    group.add_argument('--driver-buffer', default=1024, type=int, help='Size of the history buffer')
    group.add_argument('--driver-episodes', default=8, type=int, help='Number of episodes per epoch')
    group.add_argument('--driver-epochs', default=32, type=int, help='Number of epochs')
    group.add_argument('--driver-steps', default=16, type=int, help='Number of training steps per epoch')
    
    #   Group 5: Related to the post-processing
    group = arg_parser.add_argument_group('Post-process Options', 'Options for what to do after training')
    group.add_argument('--post-episode-count', default=128, type=int, help='Number of post-processing episodes to run')
    
    # Parse the arguments
    args = arg_parser.parse_args()
    run_params = args.__dict__

    # Get the host information
    host_info = get_platform_info()
    
    # Open an experiment directory
    run_hash = hashlib.sha256(json.dumps(run_params).encode()).hexdigest()[:6]
    out_dir = Path(f'runs/{args.reward}-W{args.max_size}-{datetime.utcnow().strftime("%d%m%y_%H%M%S")}-{run_hash}')  # TODO (wardlt): Do we want the time in the directory
    out_dir.mkdir(parents=True)
    
    # Save the settings
    run_params['start_time'] = datetime.utcnow().isoformat()
    with open(out_dir / 'run_params.json', 'w') as fp:
        json.dump(run_params, fp)
    with open(out_dir / 'host_info.json', 'w') as fp:
        json.dump(host_info, fp)
    
    # Make the environment
    reward, only_last = make_reward(args)
    env = SimpleEnvironment(maximum_size=args.max_size, reward=reward, only_last=only_last)
    tf_env = TFPyEnvironment(env)
    
    # Make a reward for MPNN energy
    model = tf.keras.models.load_model(args.mpnn_path, custom_objects=custom_objects)
    eng_reward = MPNNReward(model, per_water=False)
    
    # Make the GCPN 
    actor_net = GCPNActorNetwork(tf_env.observation_spec(), tf_env.action_spec(), tf_env.reset(), 
                                 num_messages=args.actor_messages,
                                 node_features=args.actor_features,
                                 graph_features=args.actor_graph_features,
                                 output_layers=3)
    critic_net = GCPNCriticNetwork(tf_env.observation_spec(), tf_env.reset(),
                                   num_messages=args.critic_messages,
                                   node_features=args.critic_features)
    
    # Use it to create the PPO agent
    #  We use clipping to avoid problems with exploding KL gradients
    tf_agent = PPOClipAgent(
        tf_env.time_step_spec(),
        tf_env.action_spec(),
        actor_net=actor_net,
        value_net=critic_net,
        optimizer=tf.keras.optimizers.Adam(args.ppo_learning_rate),
        normalize_observations=False,
        entropy_regularization=1e-4,
        discount_factor=1.,
    )
    tf_agent.initialize()
    
    # Assemble the driver
    #  TODO (wardlt): Keep a priority queue of the best structures
    buffer = TFUniformReplayBuffer(
        tf_agent.collect_data_spec,
        batch_size=1,
        max_length=args.driver_buffer,
    )
    driver = DynamicEpisodeDriver(tf_env, tf_agent.collect_policy, [buffer.add_batch], num_episodes=8)
    
    # Perform an initial test
    agent_return = compute_avg_return(tf_env, tf_agent.collect_policy, 32)
    print(f'Untrained Agent return with randomized policy: {agent_return:.1f}')
    
    # Fill the buffer
    state = tf_env.reset()
    while buffer.num_frames() < buffer.capacity:
        state, _ = driver.run(state)
        
    # Run the training loop
    train_loss = None
    first = True  # Whether we are at the first episode
    pbar = tqdm(range(args.driver_epochs))
    for epoch in pbar:
        # Compute the return of the greedy policy
        greedy_return = compute_avg_return(tf_env, tf_agent.policy, 1)
        avg_return = compute_avg_return(tf_env, tf_agent.collect_policy, 32)
        step_info = {'greedy_return': greedy_return, 'avg_return': avg_return, 'epoch': epoch}

        # Collect a few episodes using collect_policy and save to the replay buffer.
        init_ts = tf_env.reset()
        final_ts, _ = driver.run(init_ts)

        # Use data from the buffer and update the agent's network.
        dataset = buffer.as_dataset(sample_batch_size=64, num_steps=2, num_parallel_calls=4)
        for (trajs, _), step in zip(dataset, range(args.driver_steps)):
            train_loss = tf_agent.train(trajs)

            # Store step information
            step_info = {'return': avg_return, 'epoch': epoch, 'step': step}
            step_info.update(dict(zip(train_loss.extra._fields, map(float, tuple(train_loss.extra)))))
            step_info['loss'] = train_loss.loss.numpy()
           
            with open(out_dir / 'log.csv', 'a') as fp:
                writer = csv.DictWriter(fp, fieldnames=step_info.keys())
                if first:
                    first = False
                    writer.writeheader()
                writer.writerow(step_info)
                    
            # Update the progress bar
            pbar.set_description(f'loss: {train_loss.loss:.2e} - return: {avg_return:.1f} - step: {step}')
            
    # Record the after-training results
    agent_return = compute_avg_return(tf_env, tf_agent.collect_policy, 32)
    print(f'Trained Agent return with randomized policy: {agent_return:.1f}')
    
    # Save the policy
    tf_policy_saver = PolicySaver(tf_agent.collect_policy, 1)
    tf_policy_saver.save(str(out_dir / 'collect_policy'))
    
    # Do the post-processing
    rl_traj = get_trajectories(tf_env, tf_agent.collect_policy, eng_reward, args.post_episode_count)
    rl_traj.to_pickle(out_dir / 'final_trajs.pkl')
    