"""Convert data from JSON to the format used by SchNetPack"""

from multiprocessing import Pool
from schnetpack.data import AtomsData
from hydronet.data import atoms_from_dict
from argparse import ArgumentParser
from tqdm import tqdm
from glob import glob
from shutil import rmtree
import numpy as np
import gzip
import json
import os

# Parse inputs
parser = ArgumentParser()
parser.add_argument("--overwrite", action="store_true", help="Whether to overwrite existing files")
parser.add_argument("--batch-size", type=int, default=10000, help="Number of objects to write per commit")
args = parser.parse_args()

# Get the input files to read
files = glob("../data/output/geom_*.json.gz")

# Make the output directory
if os.path.isdir("data"):
    if args.overwrite:
        rmtree("data")
    else:
        raise ValueError("Output directory exists. Overwrite by adding --overwrite flag")
os.makedirs("data", exist_ok=False)

# Loop over each file
for file in files:
    # Get a name for this run
    name = os.path.basename(file)[5:-8]
    if name in ['train', 'valid']:
        data_path = "./data/train.db"
    else:
        data_path = "./data/test.db"

    # Make the output file
    data = AtomsData(data_path, available_properties=['energy'])

    # Convert every cluster in the dataset
    with Pool() as pool:
        with gzip.open(file) as fp:
            # Wrap with a progress bar
            counter = tqdm(desc=name, leave=True)

            # Iterate over chunks
            while True:
                # Check a chunk of atoms objects
                records = [json.loads(i) for i, _ in zip(fp, range(args.batch_size))]
                atoms = pool.map(atoms_from_dict, records)

                # Extract the energies
                properties = [{'energy': np.array([a.get_potential_energy()])} for a in atoms]

                # Add to the output
                data.add_systems(atoms, properties)
                counter.update(len(atoms))

                # If the batch is not batch_size, then it is because we
                #  ran out of input data
                if len(atoms) < args.batch_size:
                    break
