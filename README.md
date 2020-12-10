# HydroNet

This directory holds notebooks that illustrate how to use the HydroNet Challenge Data.

See https://exalearn.github.io/hydronet/ for more details

## Installation

Install the environment needed for these notebooks with `conda env create --file environment.yml --force`

The [`ttm`](./ttm) directory contains code needed to compute the energy of water clusters, 
which is needed for Challenge 3. 

## Quickstart

Download the data from [our Globus endpoint](https://app.globus.org/file-manager?origin_id=e38ee745-6d04-11e5-ba46-22000b92c6ec&origin_path=%2Fexalearn-design%2Fneurips%2Fdata%2Foutput%2F) to the `./data/output` folder.
The `./get-data.sh` script automates the download process if your computer has Globus Connect installed.

Each of the subdirectories hold notebooks and scripts that give example solutions to some of the challenge problems
or tutorial examples for working with HydroNet data.
