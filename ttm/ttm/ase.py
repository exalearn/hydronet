from ase.calculators.calculator import Calculator, all_changes

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
