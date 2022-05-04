import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from pytest import fixture
from pymongo import MongoClient

from hydronet.data import graph_is_valid
from hydronet.db import HydroNetRecord, HydroNetDB
from hydronet.importing import make_tfrecord
from hydronet.mpnn.data import make_data_loader


@fixture
def db() -> HydroNetDB:
    client = MongoClient()
    db = client['hydronet-pytest']
    output = HydroNetDB(db['collection'])
    output.initialize_index()
    yield output
    db.drop_collection('clusters')
    client.drop_database('hydronet-pytest')


@fixture()
def atoms():
    atoms = Atoms(numbers=[8, 1, 1] * 3,
                  positions=[[25.3875809, 2.28446364, 8.01933861],
                             [24.686451, 2.11461496, 7.36908007],
                             [26.1070786, 1.70453322, 7.77935553],
                             [22.9643402, 1.68695939, 6.75715494],
                             [22.7494984, 1.67431045, 7.70416498],
                             [22.2382431, 2.13693213, 6.33168697],
                             [23.0780773, 1.86950338, 9.5477314],
                             [22.9238548, 2.4637537, 10.2781725],
                             [23.9850082, 2.04813766, 9.2500248]])
    atoms.set_calculator(SinglePointCalculator(atoms, energy=-1))
    return atoms


def test_create(atoms):
    # Test creating from the atoms object
    record = HydroNetRecord.from_atoms(atoms)
    assert np.isclose(record.coords, atoms.get_positions()).all()
    assert record.energy == -1

    # Make sure it counted the correct number of rings
    assert record.cycle_hash == "1T0Q0P0H"

    # Make sure repeating the creation gives the same hash
    record2 = HydroNetRecord.from_atoms(atoms, -2)
    assert record2.energy == -2
    assert record.coord_hash == record2.coord_hash
    assert record.position != record2.position

    # Test JSON parsing
    json_dump = record.json()
    record3 = HydroNetRecord.parse_raw(json_dump)
    assert isinstance(record3.coords_, bytes)
    assert np.isclose(record3.coords, record.coords).all()

    # Make the atoms object
    new_atoms = record.atoms
    assert new_atoms == atoms

    # Make the coarse graph
    coarse_dict = record.coarse_dict
    assert coarse_dict['n_bonds'] == 6  # 3 bonds, counted both forward and backward
    assert graph_is_valid(record.coarse_nx, coarse=True)

    # Make the atomic graph
    atomic_dict = record.atomic_dict
    assert atomic_dict['n_bonds'] == 18  # 3 H bonds, 6 covalent

    # Make sure the TF record works for either
    make_tfrecord(record.atomic_dict)
    make_tfrecord(record.coarse_dict)


def test_db(db: HydroNetDB, atoms, tmpdir):
    # Test adding a cluster
    assert db.add_cluster(atoms, source='test')
    assert db.collection.count_documents({}) == 1

    # Make sure adding it doesn't change things
    assert not db.add_cluster(atoms)

    # ... unless we upsert
    assert db.add_cluster(atoms, upsert=True, source='real_test')

    # Loop over all records
    record = next(db.iterate_as_records(db.collection.find({})))
    assert 1 > record.position > 0
    assert record.source == 'real_test'

    # Test updating the position to shuffle the database
    orig_position = record.position
    db.shuffle()
    new_position = db.collection.find_one({})['position']
    assert orig_position != new_position

    # Write them to a TF dataset in coarse format
    path = tmpdir / 'test.protobuf'
    db.write_to_tf_records(db.collection.find({}), path)
    inputs, outputs = next(iter(make_data_loader(str(path))))
    assert np.all(outputs.numpy() == -1)
    assert np.all(inputs['n_atoms'] == 3)

    # Write them to a TF dataset in atomic format
    db.write_to_tf_records(db.collection.find({}), path, coarse=False)
    inputs, outputs = next(iter(make_data_loader(str(path))))
    assert np.all(outputs.numpy() == -1)
    assert np.all(inputs['n_atoms'] == 9)

    # Test writing the train, val, and test sets
    db.write_datasets(tmpdir)
    assert (tmpdir / 'training.proto').isfile()

    # Test writing to JSON format
    db.write_datasets(tmpdir, file_format='json')
    assert (tmpdir / 'training.json').isfile()
