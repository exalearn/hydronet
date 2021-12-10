from ase.calculators.calculator import Calculator, all_changes
from ase.optimize.optimize import Optimizer
import scipy.optimize as opt
from ttm.flib import ttm_from_f2py
import numpy as np
from ttm import TTM

class TTMCalculator(Calculator):
    """ASE interface to TTM library. 
    
    Capable of computing molecular
    
    Parameters
    ----------
    model: int
        Which version of the TTM potential to use. 
        Possible values are: 2, 21 (standing for 2.1), and 3
    """
    
    implemented_properties = ['forces', 'energy']
    default_parameters = {'model': 21}
    nolabel = True

    def calculate(
        self, atoms=None, properties=None, system_changes=all_changes,
    ):
        # Call the base class
        if properties is None:
            properties = self.implemented_properties

        Calculator.calculate(self, atoms, properties, system_changes)
        
        # Make sure out coordinates are in OHHOHH order
        z = atoms.get_atomic_numbers().tolist()
        assert all(z[i:i+3] == [8, 1, 1] for i in range(0, len(atoms), 3)), \
            "Atoms must be in OHHOHH order"
        
        # Call TTM
        ttm = TTM(self.parameters.model)
        energy, gradients = ttm.evaluate(atoms.get_positions())
        self.results['energy'] = energy
        self.results['forces'] = gradients

# Python TTM
class Converged(Exception):
    pass

class OptimizerConvergenceError(Exception):
    pass

class SciPyOptimizer(Optimizer):
    '''
    Base class for using TTM to optimize a water cluster.
    '''
    def __init__(self, atoms, logfile='-', trajectory=None,
                 callback_always=False, alpha=70.0, master=None,
                 force_consistent=None):
        restart = None
        Optimizer.__init__(self, atoms, restart, logfile, trajectory,
                           master, force_consistent=force_consistent)
        self.force_calls = 0
        self.callback_always = callback_always
        self.H0 = alpha
        self.pot_function = ttm_from_f2py
        self.model = 21

    def x0(self):
        return self.atoms.get_positions().reshape(-1)

    def f(self, x):
        self.atoms.set_positions(x.reshape(-1, 3))
        self.TTM_calc()
        return (self.energy/self.H0, self.gradients)

    def fprime(self, x):
        self.atoms.set_positions(x.reshape(-1, 3))
        self.force_calls += 1

        if self.callback_always:
            self.callback(x)

        return - self.gradients.reshape(-1,3) / self.H0

    def callback(self, x):
        self.TTM_calc()
        f = -self.gradients.reshape(-1,3)
        self.log(f)
        self.call_observers()
        if self.converged(f):
            raise Converged
        self.nsteps += 1

    def run(self, fmax=0.05, steps=100000000):
        if self.force_consistent is None:
            self.set_force_consistent()
        self.fmax = fmax
        #self.callback(None)
        try:
            self.call_fmin(fmax / self.H0, steps)
        except Converged:
            pass

    @staticmethod
    def ttm_ordering(coords):
        atom_order = []
        for i in range(0, coords.shape[0], 3):
            atom_order.append(i)
        for i in range(0, coords.shape[0], 3):
            atom_order.append(i+1)
            atom_order.append(i+2)
        return coords[atom_order,:]

    @staticmethod
    def normal_water_ordering(coords):
        atom_order = []
        Nw = int(coords.shape[0] / 3)
        for i in range(0, Nw, 1):
            atom_order.append(i)
            atom_order.append(Nw+2*i)
            atom_order.append(Nw+2*i+1)
        return coords[atom_order,:]

    def TTM_calc(self, *args):
        coords = self.ttm_ordering(self.atoms.get_positions())
        gradients, self.energy = self.pot_function(self.model, np.asarray(coords).T)
        self.gradients = self.normal_water_ordering(gradients.T).reshape(-1)

    def TTM_grad(self, *args):
        return self.gradients

    def dump(self, data):
        pass

    def load(self):
        pass

    def call_fmin(self, fmax, steps):
        raise NotImplementedError

class SciPyFminLBFGSB(SciPyOptimizer):
    """Quasi-Newton method (Broydon-Fletcher-Goldfarb-Shanno)"""
    def call_fmin(self, fmax, steps):
        output = opt.minimize(self.f, self.x0(), method='L-BFGS-B', jac=self.TTM_grad, options={'maxiter':1000, 'ftol':1e-8, 'gtol':1e-8})
