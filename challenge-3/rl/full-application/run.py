from argparse import ArgumentParser
from collections import deque
from datetime import datetime
from functools import partial, update_wrapper
from hashlib import sha256
from pathlib import Path
from threading import Event
from time import sleep
from typing import Tuple
from queue import LifoQueue, Queue, Empty
import pickle as pkl
import logging
import json
import gzip
import sys
import os

from colmena.redis.queue import ClientQueues, make_queue_pairs
from colmena.task_server import ParslTaskServer
from colmena.thinker import BaseThinker, result_processor, task_submitter, ResourceCounter, agent
from colmena.models import Result
from parsl import Config, HighThroughputExecutor
import networkx as nx

from hydronet.rl.tf.env import SimpleEnvironment
from hydronet.rl.tf.networks import GCPNActorNetwork, GCPNCriticNetwork
from hydronet.workflow import train_rl_policy, generate_clusters, invert_and_relax


def parsl_config(name: str) -> Tuple[Config, int]:
    """Make compute resource configuration
    Args:
        name: Name of the desired configuration
    Returns:
        - Parsl compute configuration
        - Number of compute slots: Includes execution slots and pre-fetch buffers
    """

    if name == 'local':
        return Config(
            executors=[
                HighThroughputExecutor(max_workers=4, prefetch_capacity=0, cpu_affinity='block')
            ],
            usage_tracking=True
        ), 4


class Thinker(BaseThinker):
    """Policy for using RL to generate potential water clusters, evaluate those clusters, and then using the new clusters to update the RL reward function

    Uses a collection of identical compute nodes where each task uses the entire node.
    """

    def __init__(self,
                 queues: ClientQueues,
                 node_count: int,
                 output_dir: Path,
                 num_to_create: int,
                 environment: SimpleEnvironment,
                 actor_net: GCPNActorNetwork,
                 critic_net: GCPNCriticNetwork,
                 queue_target: int,
                 min_generators: int
                 ):
        """

        Args:
            queues: Access to the task and result queues
            node_count: Number of nodes accessible to the agent
            output_dir: Path to the output
            num_to_create: Number of new clusters to create
            environment: Python environment used to simulate water clusters
            actor_net: Network used to pick which hydrogen bonds to add
            critic_net: Network used to value the current state of a water cluster
            queue_target: Target size of the list of clusters to be evaluated
        """
        super().__init__(queues, ResourceCounter(total_slots=node_count, task_types=['train_rl', 'generate', 'evaluate']))

        # Overall settings and state
        self.out_dir = output_dir
        self.num_to_create = num_to_create
        self.num_created = 0

        # State for the RL policy
        self.actor_net = actor_net
        self.critic_net = critic_net
        self.rl_train_step = 0
        # self.reward_fn = reward_fn
        # self.energy_mpnn = energy_mpnn
        self.env = environment

        # Task queues, and their related information
        self.eval_queue: deque[nx.DiGraph] = deque((), maxlen=int(queue_target * 1.6))  # Ensure the queue doesn't grow unbounded
        self.eval_queue_target = queue_target  # Target size of the evaluation queue
        self.min_generators = min_generators  # Minimum number of generator tasks
        self.reallocating = Event()  # Whether a re-allocation is actively in progress

        # Perform the initial resource allocation: Persistent resources for now
        self.rec.reallocate(None, 'train_rl', 1)
        self.rec.reallocate(None, 'generate', node_count - 1)

    @agent
    def train_rl(self):
        """Continually retrain the RL policy"""

        while not self.done.is_set():
            # Update reward with the latest MPNN
            # self.reward_fn.model = self.energy_mpnn

            # Update the environment with the latest reward (just in case
            # self.env.reward_fn = self.reward_fn

            # Send out the policy to be retrained
            self.queues.send_inputs(
                self.env, self.actor_net, self.critic_net,
                method='train_rl_policy',
                topic='policy',
                keep_inputs=False
            )
            self.logger.info(f'Submitted policy update round {self.rl_train_step}')

            # While that is running, save the current policy
            with gzip.open(self.out_dir / 'actor-net.pkl.gz', 'wb') as fp:
                pkl.dump(self.actor_net, fp)
            with gzip.open(self.out_dir / 'critic-net.pkl.gz', 'wb') as fp:
                pkl.dump(self.critic_net, fp)
            self.logger.info('Wrote previous state to disk')

            # Wait for the update to finish
            result = self.queues.get_result(topic='policy')
            assert result.success, result.failure_info
            self.logger.info(f'Received policy update round {self.rl_train_step}')

            # Get the results and update the state of the class
            self.actor_net, self.critic_net, train_log = result.value

            # Write the task results to disk
            with open(self.out_dir / 'policy-update-results.json', 'a') as fp:
                print(result.json(exclude={'value'}), file=fp)
            train_log.to_csv(self.out_dir / 'rl-train-log.csv', index=False)
            self.rl_train_step += 1

    @task_submitter(task_type='generate')
    def submit_generate(self):
        """Generate new clusters given the latest actor network"""

        self.logger.info(f'Submitted rollout task with actor network from round {self.rl_train_step}')
        self.queues.send_inputs(
            self.env, self.actor_net, 1000,
            method='generate_clusters',
            topic='generation',
            keep_inputs=False,
            task_info={'actor_gen': self.rl_train_step}
        )

    @result_processor(topic='generation')
    def receive_generate(self, result: Result):
        """Receive task generation, send prioritized task for evaluation"""

        # Mark that resources are now free
        self.rec.release('generate', 1)

        # Make sure it was successful
        assert result.success, result.failure_info

        # Count how many are returned
        new_clusters = result.value
        self.logger.info(f'Received {len(new_clusters)} created with actor net from round {result.task_info["actor_gen"]}')

        # Push them to the evaluation equeue
        for new in new_clusters:
            self.eval_queue.append(new)
        self.logger.info(f'New depth of evaluation queue is now {len(self.eval_queue)}')

        # If the queue is 50% over the target size, move a worker to the allocation task
        if len(self.eval_queue) >= self.eval_queue_target * 1.5 \
                and not self.reallocating.is_set() \
                and self.rec.allocated_slots("generate") > self.min_generators:
            self.logger.info('Queue has grown past target size, reallocating nodes to evaluation')
            self.reallocating.set()
            self.rec.reallocate('generate', 'evaluate', 1, block=False, callback=self.reallocating.clear)

        # Save the result to disk
        with open(self.out_dir / 'cluster-generation-results.json', 'a') as fp:
            print(result.json(exclude={'value', 'inputs'}), file=fp)

    @task_submitter(task_type='evaluate')
    def submit_evaluate(self):
        """Submit graphs to be evaluated"""

        # Get a chunk of graphs to evaluate
        chunk_size = 100
        self.logger.info(f'Gathering {chunk_size} graphs to evaluate')
        to_evaluate = []
        standoff = 10.
        while len(to_evaluate) < 100 and not self.done.is_set():
            try:
                to_evaluate.append(self.eval_queue.pop())
            except IndexError:
                self.logger.info(f'Evaluation pool is empty. Waiting for {standoff:.1f}s.')
                standoff = min(60, standoff * 1.5)  # Wait up to 1 minute for new clusters
                sleep(standoff)

        # Submit them to invert
        self.logger.info(f'Submitting {chunk_size} graphs to evaluate. Backlog: {len(self.eval_queue)}')
        self.queues.send_inputs(
            to_evaluate,
            method='invert_and_relax',
            topic='evaluation',
            keep_inputs=False
        )

    @result_processor(topic='evaluation')
    def receive_evaluate(self, result: Result):
        """Receive inverted graphs and store them to disk (later our cloud DB)"""

        # Mark that resources are now free
        self.rec.release('evaluate', 1)

        # Make sure it was successful
        assert result.success, result.failure_info

        # Save the result to disk
        with (self.out_dir / 'evaluation-records.json').open('a') as fp:
            print(result.json(exclude={'value', 'inputs'}), file=fp)

        # Write the records to disk
        new_clusters = result.value
        self.num_created += len(new_clusters)
        self.logger.info(f'Retrieved {len(new_clusters)} new water clusters. {self.num_to_create - self.num_created} left to evaluate')
        with gzip.open(self.out_dir / 'new-records.json.gz', 'at') as fp:
            for record in result.value:
                print(record.json(), file=fp)
        self.logger.info('Saved them to disk')

        # If the queue is 50% below the target size, move a worker back to generation
        if len(self.eval_queue) < self.eval_queue_target * 0.5 and self.reallocating.is_set():
            self.logger.info('Queue has shrunk to below target size, reallocating nodes to generation')
            self.reallocating.set()
            self.rec.reallocate('evaluate', 'generate', 1, block=False, callback=self.reallocating.clear)

        # Determine if we're done
        if self.num_created >= self.num_to_create:
            self.done.set()


if __name__ == '__main__':
    # Make the argument parser
    parser = ArgumentParser()

    parser.add_argument('--num-to-evaluate', help='Number of new water clusters to create and validate', type=int, default=1000)

    #  RL settings
    group = parser.add_argument_group(title='RL Options', description='Settings related to training the RL policy')
    group.add_argument('--rl-directory', help='Path to the directory containing an initial policy, environment, and reward function')

    #  Coordination between threads
    group = parser.add_argument_group(title='Coordination', description='Coordination between different types of tasks')
    group.add_argument('--evaluation-buffer', type=int, help='Target size of buffer of tasks between the generation and evaluation tasks. '
                                                             'The application will attempt to stay within 50% of this value.', default=10000)
    group.add_argument('--min-generators', type=int, default=1, help='Minimum number of workers devoted to generation tasks.')

    #  Computational resources
    group = parser.add_argument_group(title='Computational Resources', description='Resources available to the thinker')
    group.add_argument('--redis-host', help='Hostname for the redis server', default='localhost')

    # Parse the arguments
    args = parser.parse_args()
    run_params = args.__dict__
    start_time = datetime.now()

    # Check that the "rl_directory" exists
    rl_dir = Path(args.rl_directory)
    assert rl_dir.is_dir(), f'{rl_dir} does not exist'

    # Create an output directory with the name of the run
    run_hash = sha256()
    run_hash.update(str(run_params).encode())
    out_path = Path().joinpath('runs', f'{start_time.strftime("%y%m%d-%H%M%S")}-{run_hash.hexdigest()[:6]}')
    out_path.mkdir(parents=True)

    # Store the run parameters
    with open(out_path / 'run-config.json', 'w') as fp:
        json.dump(run_params, fp, indent=2)

    # Prepare the logging
    handlers = [logging.FileHandler(out_path / 'runtime.log', mode='w'), logging.StreamHandler(sys.stdout)]


    class ParslFilter(logging.Filter):
        """Filter out Parsl debug logs"""

        def filter(self, record):
            return not (record.levelno == logging.DEBUG and '/parsl/' in record.pathname)


    for h in handlers:
        h.addFilter(ParslFilter())

    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        level=logging.INFO, handlers=handlers)

    logger = logging.getLogger('main')
    logger.info(f'Created a run directory {out_path}')

    # Load in the RL components
    with (rl_dir / 'env.pkl').open('rb') as fp:
        env = pkl.load(fp)
    logger.info('Loaded environment')
    with (rl_dir / 'actor_network.pkl').open('rb') as fp:
        actor_net = pkl.load(fp)
    logger.info('Loaded actor network')
    with (rl_dir / 'critic_network.pkl').open('rb') as fp:
        critic_net = pkl.load(fp)
    logger.info('Loaded critic network')

    # Make the task queues
    client_queues, server_queues = make_queue_pairs(
        name=run_hash.hexdigest()[:12],
        hostname=args.redis_host,
        topics=['policy', 'generation', 'evaluation', 'updating'],
        serialization_method='pickle'
    )

    # Create the functions to be served with some arguments pre-defined
    my_update_policy = partial(train_rl_policy,
                               training_cycles=2,
                               buffer_size=32,
                               episodes_per_cycle=4)
    update_wrapper(my_update_policy, train_rl_policy)

    my_rollout = partial(generate_clusters)
    update_wrapper(my_rollout, generate_clusters)

    # Create the task server
    config, n_slots = parsl_config('local')
    logger.info(f'Created a Parsl server with {n_slots} workers')
    task_server = ParslTaskServer(
        queues=server_queues,
        config=config,
        methods=[my_update_policy, my_rollout, invert_and_relax]
    )

    # Create the thinker
    thinker = Thinker(
        queues=client_queues,
        node_count=n_slots,
        queue_target=args.evaluation_buffer,
        min_generators=args.min_generators,
        num_to_create=args.num_to_evaluate,
        output_dir=out_path,
        actor_net=actor_net,
        critic_net=critic_net,
        environment=env,
    )

    try:
        # Launch the servers
        #  The method server is a Thread, so that it can access the Parsl DFK
        #  The task generator is a Thread, so that all debugging methods get cast to screen
        task_server.start()
        thinker.start()
        logging.info(f'Running on {os.getpid()}')
        logging.info('Launched the servers')

        # Wait for the task generator to complete
        thinker.join()
        logging.info('Task generator has completed')
    finally:
        client_queues.send_kill_signal()

    # Wait for the method server to complete
    task_server.join()
