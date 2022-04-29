"""Workflow-friendly interfaces to key actions in water cluster optimization

We want functions that are pure and take/produce data types which are serializable.
"""
from typing import Tuple, Dict, Any, Optional

from tf_agents.agents import PPOClipAgent
from tf_agents.drivers.dynamic_episode_driver import DynamicEpisodeDriver
from tf_agents.environments import PyEnvironment, TFPyEnvironment
from tf_agents.replay_buffers import TFUniformReplayBuffer
import tensorflow as tf
import pandas as pd

from hydronet.rl.tf.networks import GCPNActorNetwork, GCPNCriticNetwork


def train_rl_policy(
        env: PyEnvironment,
        actor_net: GCPNActorNetwork,
        critic_net: GCPNCriticNetwork,
        training_cycles: int = 16,
        buffer_size: int = 1024,
        episodes_per_cycle: int = 32,
        train_steps_per_cycle: int = 16,  # TODO (wardlt): Get defaults based on what Kristina finds
        ppo_learning_rate: float = 1e-3,
        ppo_options: Optional[Dict[str, Any]] = None,
) -> Tuple[GCPNActorNetwork, GCPNCriticNetwork, pd.DataFrame]:
    """Update a reinforcement learning agent
    
    Performs a standard "rollout for an episode then update" loop on a single node.
    We first perform a series of rollout steps and then use the buffer of training steps
    to update the policy.
    
    We do not store any of the states and reward produced during training
    to reduce communication costs. Right now, the idea is that detailed analyses of the RL 
    performance are better performed outside of a larger workflow.
    
    Args:
        env: Environment that includes a reward function
        actor_net: Network used to predict which moves to take
        critic_net: Network used to assign value to a certain state
        training_cycles: Number of cycles of "rollout" and "update"
        buffer_size: Size of the training data buffer
        episodes_per_cycle: Number of episodes to run per training cycle
        train_steps_per_cycle: Number of training steps per training cycle
        ppo_learning_rate: Learning rate for training the networks
        ppo_options: Options passed to TF Agent's PPOAgent wrapper
    Returns:
        - Updated version of the actor network
        - Updated version of the critic network
        - Log of the training
    """
    # Replace default args
    if ppo_options is None:
        ppo_options = {}

    # Wrap the environment in a TFEnvironment wrapper
    tf_env = TFPyEnvironment(env)

    # Assemble the PPO agent
    tf_agent = PPOClipAgent(
        tf_env.time_step_spec(),
        tf_env.action_spec(),
        actor_net=actor_net,
        value_net=critic_net,
        optimizer=tf.keras.optimizers.Adam(ppo_learning_rate),
        normalize_observations=False,
        **ppo_options
    )
    tf_agent.initialize()

    buffer = TFUniformReplayBuffer(
        tf_agent.collect_data_spec,
        batch_size=1,
        max_length=buffer_size,
    )
    driver = DynamicEpisodeDriver(tf_env, tf_agent.collect_policy, [buffer.add_batch], num_episodes=episodes_per_cycle)

    # Fill the buffer
    state = tf_env.reset()
    while buffer.num_frames() < buffer.capacity:
        state, _ = driver.run(state)

    # Run the training loop
    train_log = []
    for epoch in range(training_cycles):
        # Collect a few episodes using collect_policy and save to the replay buffer.
        init_ts = tf_env.reset()
        final_ts, _ = driver.run(init_ts)

        # Use data from the buffer and update the agent's network.
        dataset = buffer.as_dataset(sample_batch_size=64, num_steps=2, num_parallel_calls=4)
        for (trajs, _), step in zip(dataset, range(train_steps_per_cycle)):
            train_loss = tf_agent.train(trajs)

            # Store step information
            step_info = {'epoch': epoch, 'step': step}
            step_info.update(dict(zip(train_loss.extra._fields, map(float, tuple(train_loss.extra)))))
            step_info['loss'] = train_loss.loss.numpy()
            train_log.append(step_info)

    return actor_net, critic_net, pd.DataFrame(train_log)
