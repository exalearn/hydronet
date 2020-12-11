"""Functions for inverting a structure from graph to coordinates"""

from ase.calculators.calculator import Calculator, all_changes
from ase.optimize.bfgs import BFGS
import networkx as nx
import numpy as np
from math import sqrt
import ase
import os

_h20_bond_angle = 2 * np.pi * 104.5 / 360


class HarmonicCalculator(Calculator):
    """Calculator where certain atoms are connected by a linear spring
    
    Parameters
    ----------
    graph: nx.Graph
        Bonding graph of the network
    r: float
        Desired bond length
    k: float
        Bond strength
    """
    
    implemented_properties = ['forces', 'energy']
    default_parameters = {'graph': None, 'r': 1., 'k': 1.}
    nolabel = True
    
    def calculate(
        self, atoms=None, properties=None, system_changes=all_changes,
    ):
        # Call the base class
        if properties is None:
            properties = self.implemented_properties

        Calculator.calculate(self, atoms, properties, system_changes)
        
        # Loop over all nodes in the graph
        assert self.parameters.graph is not None, "You must define the adjacency matrix"
        forces = np.zeros((len(atoms), 3))
        energy = 0
        for i, center in enumerate(atoms.positions):
            # Get the bonded atoms
            bonds = list(self.parameters.graph[i].keys())
            
            # Compute the displacement vectors and distances
            disps = atoms.positions[bonds, :] - center[None, :]
            dists = np.linalg.norm(disps, axis=1)
            energy += 0.5 * self.parameters.k * np.power(dists - self.parameters.r, 2).sum()
            
            # Compute the forces
            forces[i, :] = np.sum((self.parameters.k * (dists - self.parameters.r) / dists)[:, None] * disps, axis=0)
            
        # Store the outputs
        self.results['energy'] = energy / 2
        self.results['forces'] = forces


def convert_directed_graph_to_xyz(graph: nx.DiGraph, n_h_guesses: int = 5, 
                                  hbond_distance: float = 2.9, relax_with_harmonic: bool = False) -> ase.Atoms:
    """Generate initial coordinates for water cluster from the directed graph of bonded waters

    Args:
        graph: Directed graph of how waters are bonded
        n_h_guesses: Number of guesses to make for each hydrogen position.
            We select the one that is farthest from all other atoms
        hbond_distance: Target oxygen-to-oxygen hydrogen bond distance
        relax_with_harmonic: Whether to relax the structure with a harmonic potential
            before adding hydrogens. Experimental feature.
        
    Returns:
        Atoms object describing candidate position of hydrogens
    """
    
    # Generate coordinates using the spring model
    pos = nx.spring_layout(graph.to_undirected(), iterations=200, dim=3, center=(0, 0, 0))
    
    # Make the coordinates into a numpy array
    pos = np.array([pos[i] for i in range(len(pos))])  # From a dict
    
    # Expand so that most bonds are longer than 2.9 A
    bond_lengths = []
    for a, b in graph.edges:
        bond_lengths.append(np.linalg.norm(pos[a] - pos[b]))
    scale_factor = hbond_distance / np.percentile(bond_lengths, 25)
    pos *= scale_factor
    
    # Relax with a harmonic potential
    if relax_with_harmonic:
        calc = HarmonicCalculator(graph=graph.to_undirected(), r=hbond_distance)
        # TODO (wardlt): We likely need a simple repulsion to prevent overlap (as is used in nx.spring_layout)
        o_atoms = ase.Atoms(positions=pos)
        o_atoms.set_calculator(calc)
        opt = BFGS(o_atoms, logfile=os.devnull)
        opt.run()
        pos = o_atoms.positions.copy()
    
    # Make a map of oxygen index to covalently-bonded hydrogens
    h_map = [[] for _ in range(pos.shape[0])]
    h_count = 0
    
    # Place hydrogens 0.9641 Ang along donated bond
    h_pos = []
    for a, b, t in graph.edges(data='label'):
        if t == 'accept': 
            continue
        
        # Compute position along A->B path
        vec = pos[b] - pos[a]
        t = 0.9641 / np.linalg.norm(vec)
        
        # Clip it so that it is at least as close
        #  to the oxygen with the covalent bond
        t = min(0.5, t)
        h_pos.append(pos[a] + t * vec)
        
        # Associate this hydrogen with the oxygen
        h_map[a].append(h_count)
        h_count += 1
    pos = np.vstack((pos, h_pos))
    
    # Place remaining hydrogens
    for oxy in graph.nodes():
        # Determine the edges associated with donated hydrogen bonds
        #  Hydrogens for these are already placed
        my_h = [e for e in graph[oxy] if graph[oxy][e]['label'] == 'donate']
        
        # If you have two hydrogens, no more to place
        if len(my_h) == 2:
            # TODO (wardlt): Force the angle to be ~106.5, rotating each bond towards each other
            continue
        
        # If you have none, first place one randomly
        #  Use Gaussian-distributed points to generate random points
        #  https://mathworld.wolfram.com/SpherePointPicking.html
        if len(my_h) == 0:
            new_vecs = np.random.normal(size=(n_h_guesses, 3))
            new_vecs = 0.9641 / np.linalg.norm(new_vecs, axis=1)[:, None] * new_vecs
            possible_placements = new_vecs + pos[oxy]
            
            # Determine distances 
            #  Shape is: possible_placements x current_atom_count
            all_dists = np.linalg.norm(pos[None, :, :] - possible_placements[:, None, :], axis=-1)
            closest_dist = np.min(all_dists, axis=1)  # Shape: possible_placements
            
            # Find the farthest atom
            best_H = np.argmax(closest_dist)
            
            # Get the direction to the hydrogen we picked
            h1_dir = new_vecs[best_H, :]
            
            # Add it to the list of atoms
            #  TODO (wardlt): Is continually re-allocating the array an performance issue?
            pos = np.vstack((pos, possible_placements[best_H, :]))
            
            # Associate the hydrogen with the oxygen
            h_map[oxy].append(h_count)
            h_count += 1
            
        # If you have 1, get the direction to that hydrogen
        #  If you have 0, this should have been chosen already
        if len(my_h) == 1:
            h1_dir = my_h[0] - pos[oxy]
        h1_dir /= np.linalg.norm(h1_dir)
            
        # Make a bunch of vectors that are orthogonal to the target direction
        rand_ys = np.random.normal(size=(n_h_guesses, 3))
        rand_ys = rand_ys / np.linalg.norm(rand_ys, axis=1)[:, None]
        rand_ys = np.cross(h1_dir, rand_ys)
        rand_ys = rand_ys / np.linalg.norm(rand_ys, axis=1)[:, None]
        
        # Using the plane defined by h1_dir and each new "y", 
        #  Compute the point that is a 104.5 degree rotation
        new_vecs = np.cos(_h20_bond_angle) * h1_dir[None, :] + np.sin(_h20_bond_angle) * rand_ys
        
        # Compute new coordinates for the H that are 
        #  0.9641 Ang along that direction
        possible_placements = 0.9641 * new_vecs + pos[oxy]
        
        # Determine distances 
        #  Shape is: possible_placements x current_atom_count
        all_dists = np.linalg.norm(pos[None, :, :] - possible_placements[:, None, :], axis=-1)
        closest_dist = np.min(all_dists, axis=1)  # Shape: possible_placements
        
        # Find the farthest atom
        best_H = np.argmax(closest_dist)
        
        # Add it to the list of atoms
        pos = np.vstack((pos, possible_placements[best_H, :]))
        
        # Associate the hydrogen with the oxygen
        h_map[oxy].append(h_count)
        h_count += 1
        
    # Re-order into OHHOHH format
    #  We have been keeping track of which Hs are bonded to which other 
    new_order = np.zeros(pos.shape[:1], dtype=np.int)
    for i, h_s in enumerate(h_map):  # Loop over each oxygen
        assert len(h_s) == 2
        # O's are in the first third. Need to offset index for Hs
        h_offset = np.add(h_s, len(graph))  
        new_order[i * 3] = i
        new_order[i*3+1:i*3+3] = h_offset
    pos = pos[new_order, :]
    
    # Build the ase.Atoms object
    symbols = ['O', 'H', 'H'] * len(graph)
    atoms = ase.Atoms(symbols=symbols, positions=pos)
    
    return atoms





def rigid_transform_3D(A, B):
    # Input: expects 3xN matrix of points
    # Returns R,t
    # R = 3x3 rotation matrix
    # t = 3x1 column vector
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        #print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    t = -R@centroid_A + centroid_B

    return R, t


def RMSE_align(A, B, n=3):
    ''' 
    Given two sets of 3D points and their correspondence the algorithm will return a 
    least square optimal rigid transform (also known as Euclidean) between the two sets. 
    The transform solves for 3D rotation and 3D translation, no scaling.

    Amended from: https://github.com/nghiaho12/rigid_transform_3D
    "Least-Squares Fitting of Two 3-D Point Sets", Arun, K. S. and Huang, T. S. and Blostein, S. D, 
    IEEE Transactions on Pattern Analysis and Machine Intelligence, Volume 9 Issue 5, May 1987  
    '''
    # Recover R and t
    A = np.array(A).T
    B = np.array(B).T
    ret_R, ret_t = rigid_transform_3D(A, B)

    # Compare the recovered R and t with the original
    B2 = (ret_R@A) + ret_t

    # Find the root mean squared error
    err = B2 - B
    err = err * err
    err = np.sum(err)
    rmse = np.sqrt(err/n)
    return rmse
