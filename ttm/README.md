# TTM Potential for Python

This directory contains a Python wrapper for the TTM iteraction potential for water.

The wrapper was originally developed by Joseph Heindal and packaged for distribution by Logan Ward.

## Requirements

You must have Python with numpy installed and a Fortran compiler.

The installation instructions (below) assume that you have gFortran, 
though it should be possible to install with other Fortran compilers
by editing [`Makefile.ttm`](./Makefile.ttm).

## Installation

The firs step is to download and build the TTM library from PNNL.
If you are running Linux and have gFortran, calling

```bash
./get-ttm.sh
```

will download and build the library.

Once the library is compiled, install the package with 

```bash
pip install -e .
```

## Usage

We recommend using the TTM potential through the [ASE Calculator](https://wiki.fysik.dtu.dk/ase/ase/calculators/calculators.html)
interface provided by this package.

```python
from ttm.ase import TTMCalculator
import ase

atoms = ase.Atoms()
calc = TTMCalculator()
energy = calc.get_potential_energy(atoms)
```

We also provide an implementation for geometry relaxation using 
the TTM potential and the scipy.optimize.minimize function with
an ase wrapper. Running the dynamics updates the ase Atoms
object, can then be passed to the TTMCalculator.


```python
from ttm.ase import SciPyFminLBFGSB
import ase

atoms = ase.Atoms()
dyn = SciPyFminLBFGSB(atoms)
dyn.run(fmax=1e-2)
```
