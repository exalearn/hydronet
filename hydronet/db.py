"""Tools related to storing energies in a database"""
import hashlib
import random
import base64
import pickle as pkl
from pathlib import Path
from datetime import datetime
from typing import Optional, Iterable, Union

import ase
import networkx as nx
import numpy as np
import tensorflow as tf
from ase.calculators.singlepoint import SinglePointCalculator
from pymongo import MongoClient
from pymongo.collection import Collection
from pydantic import BaseModel, Field, validator
from pymongo.cursor import Cursor
from pymongo.errors import DuplicateKeyError

from hydronet.data import graph_from_dict
from hydronet.descriptors import count_rings
from hydronet.importing import create_graph, coarsen_graph, create_inputs_from_nx, make_tfrecord


class HydroNetRecord(BaseModel):
    """Record holding data about a single water cluster.
    
    For simplicity in updated the database, we use a 
    """

    # Geometry information
    n_waters: int = Field(..., description='Number of water molecules in this cluster')
    coords_: bytes = Field(None, description='XYZ coordinates of water and clusters in OHHOHH order. Stored as a binary string')

    # Energy of the cluster
    energy: float = Field(..., description='TTM-computed energy of the molecule')

    # Information used to reconstruct the graph form of the cluster
    coarse_bond_: bytes = Field(None, description='Bond types for the coarse graph')
    coarse_connectivity_: bytes = Field(None, description='Connectivity for the coarse graph')
    atomic_bond_: bytes = Field(None, description='Bond types for the coarse graph')
    atomic_connectivity_: bytes = Field(None, description='Connectivity for the coarse graph')
    
    # Details about the graph structure
    cycle_hash: str = Field(..., description='List of the number of cycles of between 3 and 6, written as a string.')

    # Useful tools for the database
    position: float = Field(default_factory=lambda: random.random(), description='Random number used to assign data to training/validation/test sets')
    coord_hash: Optional[str] = Field(None, min_length=64, max_length=64,
                                      description='Hash of the coordinates. Used to ensure structures are not exact duplicates')
    graph_hash: Optional[str] = Field(None, max_length=32, min_length=32, description='Weisfeiler Lehman (WL) hash of the ')

    # Useful tools for provenance
    create_date: datetime = Field(default_factory=datetime.now, description='Time record was created or updated')
    source: Optional[str] = Field(None, description='Where this entry came from')
    
    class Config:
        json_encoders = {bytes: lambda x: '_base64' + base64.b64encode(x).decode('ascii')}
        
    @validator('coords_', 'coarse_bond_', 'coarse_connectivity_', 'atomic_bond_', 'atomic_connectivity_')
    def _debase64_encode(v):
        if v[:7] == b'_base64': # Attempt to decode base64
            return base64.b64decode(v[7:])
        return v

    def __init__(self, **kwargs):
        """Not to be used by most users."""
        super().__init__(**kwargs)
        if 'coords_' in kwargs:
            sha = hashlib.sha256()
            sha.update(self.coords_)
            self.coord_hash = sha.hexdigest()[:64]
        elif 'coord_hash' not in kwargs:
            raise ValueError('You must either provide coords_ or coord_hash')
        if 'atomic_bond_' in kwargs or 'coarse_bond_' in kwargs:
            self.graph_hash = nx.algorithms.weisfeiler_lehman_graph_hash(self.atomic_nx, edge_attr='label', node_attr='label',
                                                                         iterations=nx.diameter(self.atomic_nx) + 1)
        elif 'graph_hash' not in kwargs:
            raise ValueError('You must either provide a graph or graph_hash')

    def __repr__(self):
        return f'n_waters={self.n_waters} energy={self.energy}'

    def __str__(self):
        return f'n_waters={self.n_waters} energy={self.energy}'

    @property
    def coords(self) -> np.ndarray:
        """Copy of the XYZ coordinate array"""
        return pkl.loads(self.coords_)

    @property
    def z(self) -> np.ndarray:
        """Atomic numbers of each atom"""
        return np.array([8, 1, 1] * self.n_waters, dtype=np.int64)

    @property
    def atom(self) -> np.ndarray:
        """Types of each atom in the array"""
        return np.array([0, 1, 1] * self.n_waters, dtype=np.int64)

    @property
    def atoms(self) -> ase.Atoms:
        # Make the atoms object
        atoms = ase.Atoms(positions=self.coords, numbers=self.z)

        # Attach the energy value
        calc = SinglePointCalculator(atoms, energy=self.energy)
        atoms.set_calculator(calc)

        return atoms

    @property
    def coarse_dict(self) -> dict:
        """Dictionary representation of the coarse graph"""
        conn = pkl.loads(self.coarse_connectivity_)
        return {
            'n_waters': self.n_waters,
            'n_atoms': self.n_waters,
            'n_bonds': len(conn),
            'atom': np.zeros((self.n_waters,), dtype=np.int64),
            'bond': pkl.loads(self.coarse_bond_),
            'connectivity': conn,
            'energy': self.energy
        }

    @property
    def coarse_nx(self) -> nx.DiGraph:
        """NetworkX version of the coarse graph"""
        return graph_from_dict(self.coarse_dict)

    @property
    def atomic_dict(self) -> dict:
        """Dictionary representation of the atomic graph"""
        conn = pkl.loads(self.atomic_connectivity_)
        return {
            'n_waters': self.n_waters,
            'n_atoms': self.n_waters * 3,
            'n_bonds': len(conn),
            'atom': self.atom,
            'bond': pkl.loads(self.atomic_bond_),
            'connectivity': conn,
            'energy': self.energy
        }

    @property
    def atomic_nx(self) -> nx.Graph:
        """NetworkX version of the atomic graph"""
        return graph_from_dict(self.atomic_dict)

    @classmethod
    def from_atoms(cls, atoms: ase.Atoms, energy: Optional[float] = None) -> 'HydroNetRecord':
        """Create a new database entry from a water

        Stores the connectivity information as well

        Args:
            atoms: 3D geometry of the water cluster
            energy: Energy of the structure, computed with TTM
        Returns:
            Database entry for this cluster, complete with the graph data
        """

        # If energy is not provided, get it from the structure
        if energy is None:
            energy = atoms.get_potential_energy()

        # Extract the number of waters and the coordinates
        coords_ = atoms.get_positions().dumps()
        n_waters = len(atoms) // 3

        # Get an atomic and coarse graph
        atomic_graph = create_graph(atoms)
        coarse_graph = coarsen_graph(atomic_graph)

        # Store the bond and connectivity information as numpy arrays
        atomic_dict = create_inputs_from_nx(atomic_graph)
        atomic_bond_ = np.array(atomic_dict["bond"]).dumps()
        _atomic_conn = np.array(atomic_dict["connectivity"]).dumps()

        coarse_dict = create_inputs_from_nx(coarse_graph)
        coarse_bond_ = np.array(coarse_dict["bond"]).dumps()
        _coarse_conn = np.array(coarse_dict["connectivity"]).dumps()
        
        # Count the number of cycles
        cycle_hash = "".join(
            f"{count_rings(coarse_graph, i)}{l}"
            for i, l in zip([3, 4, 5, 6], ['T', 'Q', 'P', 'H'])
        )

        # Initialize the object
        return cls(
            n_waters=n_waters, coords_=coords_,
            energy=energy,
            atomic_bond_=atomic_bond_, atomic_connectivity_=_atomic_conn,
            coarse_bond_=coarse_bond_, coarse_connectivity_=_coarse_conn,
            cycle_hash=cycle_hash
        )


class HydroNetDB:
    """Wrapper for a MongoDB holding properties of water clusters"""

    def __init__(self, collection: Collection):
        """
        Args:
            collection: Collection of molecule property documents
        """
        self.collection = collection

    @classmethod
    def from_connection_info(cls, hostname: str = "localhost", port: Optional[int] = None,
                             database: str = "hydronet", collection: str = "clusters", **kwargs) -> 'HydroNetDB':
        """Connect to MongoDB and create the database wrapper

        Args:
            hostname: Host of the MongoDB
            port: Port of the service
            database: Name of the database holding desired data
            collection: Name of the collection holding the water cluster data
        """
        client = MongoClient(hostname, port=port, **kwargs)
        db = client.get_database(database)
        return cls(db.get_collection(collection))

    def initialize_index(self):
        """Prepare a new collection.

        Makes the "coord_hash" a unique key, sorts on n_waters and position
        """
        self.collection.create_index('coord_hash', unique=True)
        self.collection.create_index('graph_hash')
        self.collection.create_index([('position', 1)])
        self.collection.create_index([('n_waters', 1)])
        return

    def add_cluster(self, atoms: ase.Atoms, energy: Optional[float] = None, upsert: bool = False, source: Optional[str] = None) -> bool:
        """Add a cluster to the database

        Args:
            atoms: Atomic cluster of interest
            energy: Energy of cluster in kcal/mol. If not provided, will be extracted from ``atoms.get_potential_energy()``
            upsert: Whether to update an existing record
            source: Source for the structure
        Returns:
            Whether the database was updated
        """
        record = HydroNetRecord.from_atoms(atoms, energy)
        record.source = source

        if upsert:
            key = record.coord_hash
            self.collection.update_one({'coord_hash': key}, {'$set': record.dict()}, upsert=True)
            return True
        else:
            try:
                self.collection.insert_one(record.dict())
                return True
            except DuplicateKeyError:
                return False

    def shuffle(self):
        """Assign a new order to the atoms"""
        self.collection.update_many({}, [{'$set': {'position': {'$rand': {}}}}])

    @staticmethod
    def iterate_as_records(cursor: Cursor) -> Iterable[HydroNetRecord]:
        """Iterate through a series of records as HydroNet records

        Args:
            cursor: Cursor that will iterate over the database
        Yields:
            Records in the order presented
        """

        for data in cursor:
            yield HydroNetRecord.parse_obj(data)

    def write_to_tf_records(self, cursor: Cursor, path: Union[str, Path], coarse: bool = True):
        """Write results of a query to TF protobuf format

        Args:
            cursor: Results of a query
            path: Path to the TF record object
            coarse: Whether to write the coarse or atomic graph
        """

        options = tf.io.TFRecordOptions(
            compression_type="ZLIB",
            compression_level=5,
        )
        with tf.io.TFRecordWriter(str(path)) as out:
            for record in self.iterate_as_records(cursor):
                dict_format = record.coarse_dict if coarse else record.atomic_dict
                out.write(make_tfrecord(dict_format))

    def write_datasets(self, directory: Union[str, Path], coarse: bool = True, val_split: float = 0.1, test_split: float = 0.1):
        """Write the training, test, and validation sets to a directory

        Args:
            directory: Path to an output directory
            coarse: Whether to write the coarse or atomic graphs
            val_split: Fraction of data to use for the validation set
            test_split: Fraction of the data to use for the test set
        """

        assert val_split + test_split < 1, 'Training and test sets should add to less than 1'

        # Make the directory, if need be
        output_dir = Path(directory)
        output_dir.mkdir(exist_ok=True, parents=True)

        # Save the training set
        project = {'coords_': False}
        if coarse:
            project.update({
                'atomic_bond_': False,
                'atomic_connectivity_': False,
            })
        else:
            project.update({
                'coarse_bond_': False,
                'coarse_connectivity_': False
            })
        cursor = self.collection.find({'$and': [{'position': {'$gt': val_split}},
                                                {'position': {'$lt': 1 - test_split}}]},
                                     projection=project).sort('position')
        self.write_to_tf_records(cursor, output_dir / 'training.proto', coarse=coarse)

        # Save the validation set
        cursor = self.collection.find({'position': {'$lt': val_split}}, projection=project).sort('position')
        self.write_to_tf_records(cursor, output_dir / 'validation.proto', coarse=coarse)

        # Save the test set
        cursor = self.collection.find({'position': {'$gt': 1 - test_split}}, projection=project).sort('position')
        self.write_to_tf_records(cursor, output_dir / 'test.proto', coarse=coarse)
