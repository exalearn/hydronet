import sys
import numpy as np

class TTM:
    def __init__(self, model=21):
        """Evaluates the energy and gradients of the TTM family of potentials.

        Args:
            model (int, optional): The TTM model which will be used. Options are 2, 21, and 3. Defaults to 21.
        """
        try:
            from ttm.flib import ttm_from_f2py
        except ImportError:
            print("Could not load the ttm module. Make sure the ttm library can be linked against and the f2py module can be imported from this directory.")
            sys.exit(1)
        self.pot_function = ttm_from_f2py
        self.model = model
        possible_models = [2, 21, 3]
        if self.model not in possible_models:
            print("The possible TTM versions are 2, 21, or 3. Please choose one of these.")
            sys.exit(1)
    
    def evaluate(self, coords):
        """Takes xyz coordinates of water molecules in O H H, O H H order and re-orders to OOHHHH order
        then transposes to fortran column-ordered matrix and calls the TTM potential from an f2py module.


        Args:
            coords (ndarray3d): xyz coordinates of a system which can be evaluated by this potential.
        Returns:
            energy (float): energy of the system in hartree
            forces (ndarray3d): forces of the system in hartree / bohr
        """
        #Sadly, we need to re-order the geometry to TTM format which is all oxygens first.
        coords = self.ttm_ordering(coords)
        gradients, energy = self.pot_function(self.model, np.asarray(coords).T)
        return energy, (-self.normal_water_ordering(gradients.T))
    
    @staticmethod
    def ttm_ordering(coords):
        """Sorts an array of coordinates in OHHOHH format to OOHHHH format.

        Args:
            coords (ndarray3d): numpy array of coordinates

        Returns:
            ndarray3d: numpy array of coordinate sorted according to the order TTM wants.
        """
        atom_order = []
        for i in range(0, coords.shape[0], 3):
            atom_order.append(i)
        for i in range(0, coords.shape[0], 3):
            atom_order.append(i+1)
            atom_order.append(i+2)
        return coords[atom_order,:]
    
    @staticmethod
    def normal_water_ordering(coords):
        """Sorts an array of coordinates in OOHHHH format to OHHOHH format.

        Args:
            coords (ndarray3d): numpy array of coordinates

        Returns:
            ndarray3d: numpy array of coordinate sorted in the normal way for water.
        """
        atom_order = []
        Nw = int(coords.shape[0] / 3)
        for i in range(0, Nw, 1):
            atom_order.append(i)
            atom_order.append(Nw+2*i)
            atom_order.append(Nw+2*i+1)
        return coords[atom_order,:]

if __name__ == '__main__':
    ttm21f = TTM()
    cage_hexamer = np.array([[0.87715956, 1.70409266, 0.47858616],
                            [1.70199937, 1.19722607, 0.29220265],
                            [1.16181787, 2.60891668, 0.65546661],
                            [-0.82054445, 0.61636011, -1.63123430],
                            [-0.26682713, 1.17302621, -1.05309865],
                            [-0.36885969, -0.24447946, -1.58195952],
                            [-0.64071556, -0.48854013, 1.64104190],
                            [-0.19915392, 0.37247798, 1.52856380],
                            [-1.54433112, -0.32408724, 1.29835600],
                            [0.57630762, -1.69154092, -0.42866929],
                            [0.43417462, -2.64420081, -0.48595869],
                            [0.08911917, -1.38946090, 0.38589939],
                            [2.81159896, -0.10395953, -0.18689424],
                            [3.46294039, -0.45173380, 0.43482057],
                            [2.15849520, -0.82915775, -0.30211113],
                            [-2.88801599, -0.06633690, 0.05761427],
                            [-2.29024563, 0.26724676, -0.65069518],
                            [-3.66545029, 0.50362404, 0.03495814]])
    energy, gradients = ttm21f.evaluate(cage_hexamer)
    print("Calculated Outputs:")
    print(energy)
    print(gradients)


    '''
    These are the output you get if you don't convert the units from kcal/mol to hartree
    or the gradients from kcal/mol/A to hartree/bohr.

    Energy (kcal/mol) =   -44.009900
    
     ----------Derivatives (kcal/mol/A)  ---------------
      1     -17.171631     -8.501426     -1.129756
      1      15.737880     -5.080882     -0.557946
      1       4.148115     11.227446      2.146692
      2     -14.176072      2.584080     -5.868841
      2       7.047068      4.271805      4.834894
      2       6.021064     -7.268982      0.103362
      3       9.904202    -16.366005      1.193872
      3       0.472663     10.252552      0.000899
      3      -9.915398      6.602982     -1.776479
      4      11.645372     12.264258    -15.884264
      4      -1.032711    -12.792524     -0.344699
      4     -12.280451      4.581854     19.976125
      5      -3.474920     15.323084     -8.571923
      5       6.292993     -4.201467      7.860706
      5      -4.973061    -12.842316     -0.098187
      6       3.244097    -13.576127      8.849190
      6       7.953128      6.482030    -11.249730
      6      -9.442338      7.039638      0.516085
    '''
