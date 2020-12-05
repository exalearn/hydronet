# Challenge 3:  Generating Low-Energy Clusters

The number of possible water clusters explodes combinatorially with the number of water molecules. 
Finding the lowest-energy structures in this space is a grand challenge in science.

Here we provide a script to evaluate generated clusters. The file features.csv must be downloaded to the groundtruth folder.


### Running the script
eval.py supports two formats for your data:
* xyz format with coordinates in Angstroms
* graph format as given in the Hydronet dataset

#### xyz format
```
python eval.py -data yourdata.xyz -size 10 -stat ks
```

#### Graph format
```
python eval.py -data yourdata.json.gz -datatype graph -size 10 -stat ks
```


### Supported metrics

The following four metrics are supported using flag -stat.

#### Kullback-Leibler divergence
KL divergence is a method of measuring statistical distance, sometimes referred to as relative entropy. 
The lower the KL divergence value, the closer the two distributions are to one another.
KL divergence is directional, such that calculating the divergence for distributions *P* and *Q* would give a different score from *Q* and *P*.
Because the test and truth distributions may be of different sizes, we implement a binning strategy to compute KL divergence. 
The default number of bins is 100, but may be changes with the -bins flag.
```
-stat kl
```

#### Jensen-Shannon divergence
JS divergence extends KL divergence to calculate a normalized symmetrical score. Therefore, the divergence of *P* from *Q* is the same as *Q* from *P*.
The -bins flag is also used here.
```
-stat js
```

#### Kolmogorov-Smirnov statistic 
The KS test is used to decide if a sample comes from a population with a specific distribution. 
The KS statistic is high when the fit is good and low when the fit is poor.
```
-stat ks
```

#### Wasserstein distance
This distance is also known as the earth mover’s distance, since it can be seen as the minimum amount of work 
required to transform *P* into *Q*, where work is measured as the amount of distribution weight that must be moved, 
multiplied by the distance it has to be moved. A lower value means the two distributions are more similar.
```
-stat wd
```




### Additional property evaluation
eval.py also supports comparison against graph properties:
* Number of trimers, tetramers, pentamers, and hexamers
* Average shortest path length
* Wiener index

Use the --graph-properties flag to turn this feature on. It works for either input type.
