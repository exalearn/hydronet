"""Trains the model using Keras without any kind of data-parallel training"""

from hydronet.mpnn.callbacks import EpochTimeLogger, LRLogger
from hydronet.mpnn.data import make_data_loader
from hydronet.mpnn.layers import build_fn
from hydronet.utils import get_platform_info

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, CSVLogger, TerminateOnNaN, EarlyStopping

from argparse import ArgumentParser
from datetime import datetime
from time import perf_counter
import pandas as pd
import numpy as np
import hashlib
import json
import os

# Hard-coded stuff
_energy_per_water = -10.520100056495238


if __name__ == "__main__":
    # Parse the arguments
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--network-choice', default='coarse', choices=['atomic', 'coarse'], 
                            help='Whether to use the coarsened or atomic graph')
    arg_parser.add_argument('--batch-size', '-b', default=256, help='Batch size', type=int)
    arg_parser.add_argument('--batch-per-epoch', default=None, help='Maximum number of batches per epoch', type=int)
    arg_parser.add_argument('-r', '--shuffle-buffer-size', help='Size of the buffer used for the shuffling',
                            type=int, default=32768)
    arg_parser.add_argument('--parallel-loader', '-p', help='Number of threads to use in data loader steps', type=int,
                            default=tf.data.experimental.AUTOTUNE)
    arg_parser.add_argument('-f', '--features', help='Number of features per node/edge', default=64, type=int)
    arg_parser.add_argument('-d', '--dropout', default=0.0, help='Dropout rate', type=float)
    arg_parser.add_argument('-o', '--output-layers', default=(64, 64, 32), help='Number of units in output hidden layers', 
                            nargs='*', type=int)
    arg_parser.add_argument('-a', '--activation', default='sigmoid', help='Activation used in message layers')
    arg_parser.add_argument('-t', '--message-steps', default=4, help='Number of message passing steps', type=int)
    arg_parser.add_argument('-s', '--learning-rate-start', default=1e-3, help='Learning rate start', type=float)
    arg_parser.add_argument('-e', '--learning-rate-end', default=1e-4, help='Learning rate end', type=float)
    arg_parser.add_argument('-n', '--epochs', help='Number of epochs to run', type=int, default=16)
    arg_parser.add_argument('-c', '--grad-clip', default=1.0, help='Gradient clipping', type=float)

    # Parse the arguments
    args = arg_parser.parse_args()
    run_params = args.__dict__
    run_params['start_time'] = datetime.utcnow().isoformat()

    # Get the host information
    host_info = get_platform_info()
    
    # Hard-coded paths for data and model
    train_path = os.path.join('..', '..', 'data', 'output', f'{args.network_choice}_train.proto')
    valid_path = os.path.join('..', '..', 'data', 'output', f'{args.network_choice}_valid.proto')
    test_path = os.path.join('..', '..', 'data', 'output', f'{args.network_choice}_test.proto')

    # Open an experiment directory
    run_hash = hashlib.sha256(json.dumps(run_params).encode()).hexdigest()[:6]
    out_dir = f'{host_info["hostname"]}-T{args.message_steps}-f{args.features}-N{args.epochs}-{run_hash}'
    os.makedirs(out_dir)

    # Save the parameters and host information
    with open(os.path.join(out_dir, 'host_info.json'), 'w') as fp:
        json.dump(host_info, fp, indent=2)
    with open(os.path.join(out_dir, 'run_params.json'), 'w') as fp:
        json.dump(run_params, fp, indent=2)

    # Make the data loader
    loader = make_data_loader(train_path, batch_size=args.batch_size,
                              shuffle_buffer=args.shuffle_buffer_size,
                              cache=True, n_threads=args.parallel_loader).prefetch(16)
    val_loader = make_data_loader(valid_path, batch_size=args.batch_size,
                                  cache=True, n_threads=args.parallel_loader).prefetch(16)
    test_loader = make_data_loader(test_path, batch_size=args.batch_size,
                                  cache=False, n_threads=args.parallel_loader).prefetch(16)
    
    # Build the model
    node_mean = _energy_per_water if args.network_choice == 'coarse' else _energy_per_water / 3
    model = build_fn(node_mean, atom_features=args.features, message_steps=args.message_steps,
                     message_layer_activation=args.activation, output_layer_sizes=args.output_layers,
                     dropout=args.dropout)
    
    # Record the size of the datasets
    def get_size(path, batch_size):
        start_time = perf_counter()
        loader = tf.data.TFRecordDataset(path).batch(batch_size).prefetch(8)
        for count, _ in enumerate(loader):
            pass
        read_time = perf_counter() - start_time
        return count, read_time

    with open(os.path.join(out_dir, 'batch_counts.json'), 'w') as fp:
        val_steps, val_time = get_size(valid_path, args.batch_size)
        train_steps, train_time = get_size(train_path, args.batch_size)
        json.dump({
            'train_size': train_steps,
            'train_time': train_time,
            'val_size': val_steps,
            'val_time': val_time
        }, fp, indent=2)

    # Compile the model
    model.compile(Adam(args.learning_rate_start, clipnorm=args.grad_clip), loss='mean_squared_error')

    # Assemble the learning rate decay function
    decay_per_epoch = np.power(args.learning_rate_end / args.learning_rate_start, 1.0 / args.epochs)

    def lr_schedule(i, lr):
        return lr * decay_per_epoch

    # Make the callbacks to use training
    callbacks = [
        EpochTimeLogger(),
        LRLogger(),
        CSVLogger(os.path.join(out_dir, 'log.csv')),
        ModelCheckpoint(os.path.join(out_dir, 'checkpoint.h5')),
        ModelCheckpoint(os.path.join(out_dir, 'best_model.h5'), save_best_only=True),
        TerminateOnNaN(),
        LearningRateScheduler(lr_schedule),
        EarlyStopping(restore_best_weights=True, patience=args.epochs // 8)
    ]

    # Fit the model
    model.fit(loader, epochs=args.epochs, shuffle=False, steps_per_epoch=args.batch_per_epoch,
              callbacks=callbacks, validation_data=val_loader, validation_steps=val_steps, verbose=1)
    
    # Run the model on the test set
    y_pred = model.predict(test_loader)
    y_true = []
    n_waters = []
    for batch in test_loader:
        n_waters.append(batch[0]['n_atom'] // 3 if args.network_choice == 'atomic' else batch[0]['n_atom'])
        y_true.append(batch[1])
    y_true = np.concatenate(y_true, axis=0)[:, 0]
    n_waters = np.hstack(n_waters)
    
    # Save the output
    results = pd.DataFrame({
        'n_waters': n_waters,
        'y_true': y_true,
        'y_pred': y_pred[:, 0]
    })
    results.to_csv(os.path.join(out_dir, 'test_predictions.csv'), index=False)
