# HydroNet

This directory holds notebooks that illustrate how to use the HydroNet Challenge Data.

See https://exalearn.github.io/hydronet/ for more details.


## Installation

Install the environment needed for these notebooks with `conda env create --file environment.yml --force`

The [`ttm`](./ttm) directory contains code needed to compute the energy of water clusters, 
which is needed for Challenge 3. 

## Quickstart

Download the data from [our Globus endpoint](https://app.globus.org/file-manager?origin_id=f10a69a9-338c-4e5b-baa1-0dc92359ab47&origin_path=%2Fexalearn-design%2Fneurips%2Fdata%2Foutput%2F) to the `./data/output` folder.
The `./get-data.sh` script automates the download process if your computer has Globus Connect installed.

Each of the subdirectories hold notebooks and scripts that give example solutions to some of the challenge problems
or tutorial examples for working with HydroNet data.

## Citation
If you find this work useful, please cite our publication: 
[Sutanay Choudhury, Jenna A. Bilbrey, Logan Ward, Sotiris S. Xantheas, Ian Foster, Joseph P. Heindel, Ben Blaiszik, Marcus E. Schwarting, "HydroNet: Benchmark Tasks for Preserving Intermolecular Interactions and Structural Motifs in Predictive and Generative Models for Molecular Data", 2020, Machine Learning and the Physical Sciences Workshop at the 34th Conference on Neural Information Processing Systems (NeurIPS).](https://arxiv.org/abs/2012.00131)
```bibtex
@article{choudhury2020hydronet,
  author    = {Sutanay Choudhury and
               Jenna A. Bilbrey and
               Logan T. Ward and
               Sotiris S. Xantheas and
               Ian T. Foster and
               Joseph P. Heindel and
               Ben Blaiszik and
               Marcus E. Schwarting},
  title     = {HydroNet: Benchmark Tasks for Preserving Intermolecular Interactions
               and Structural Motifs in Predictive and Generative Models for Molecular
               Data},
  journal   = {Machine Learning and the Physical Sciences Workshop at the 34th 
               Conference on Neural Information Processing Systems (NeurIPS)},
  volume    = {abs/2012.00131},
  year      = {2020},
  url       = {https://arxiv.org/abs/2012.00131},
  archivePrefix = {arXiv},
}
```
