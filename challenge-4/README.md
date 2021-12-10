# Challenge 4: Producing 3D Coordinates from Bonding Grpahs

Graphs may be a natural form for interpreting and generating new water clusters, but the actual object they describe is a 3D arrangements of waters.
This challenge addresses a core issue in being able to use graph generators to find optimal graph clusters: producing 3D coordinates from a bonding graph.

## Data Source

We have identified up to 100 clusters of each water cluster size in our database: 50 of the lowest-energy clusters and 50 that were selected randomly from the higher-energy clusters.

The clusters are selected from the HydroNet test set and saved in `benchmark_clusters.json`. 
The file is saved using version control, but you can also check that you are working with the correct file using `md5sum -c benchmark.md5`.

## Challenge Problems

The challenge is to turn the water cluster into a graph and then restore the 3D coordinates with optimal accuracy.

Metrics TBD but will include: RMSD between original and reproduced structure, energy difference between original and reproduced, and whether the graphs are isometric.