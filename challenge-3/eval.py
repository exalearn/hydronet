import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from utils import graph_loader, projected_graph, rings, metrics, evaluate
from tabulate import tabulate
import sys
from hydronet.data import graph_from_dict
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-data', required=True, type=str, help="Test data.")
parser.add_argument('-gt', default='./groundtruth/features.csv', type=str, help="Ground truth data for comparison.")
parser.add_argument('-datatype', default='xyz', type=str, help="Data type: xyz or graph")
parser.add_argument('-size', default=0, type=int, help="Cluster size to analyze")
parser.add_argument('-stat', default='ks', type=str, help="kl or js divergence; ks statistic; wd for Wasserstein distance")
parser.add_argument('-bins', default=100, type=int, help="Number of bins used to compute the kl/js divergence.")
parser.add_argument('--graph-properties', action='store_true', default=False, help="Compute graph properties and compare distribution.")
args = parser.parse_args()


# load ground truth data for comparison
df_comp = pd.read_csv(args.gt)

# load test data
# if 'xyz' all clusters must be the same size
# if 'graph' cluster size must be chosen with -size flag
if args.datatype == 'xyz':
    cluster_dist, energy_dist = graph_loader.read_lines_base(args.data)
    print(str(len(cluster_dist))+' clusters in distribution\n')
    # check for only one cluster size in test data
    if args.size*3 != np.array([len(x)-2 for x in cluster_dist]).mean():
        sys.exit('Multiple cluster sizes found in test data. Please split test data into different files based on cluster size.')

elif args.datatype == 'graph':
    if args.size == 0:
        sys.exit('Use -size flag to choose the cluster size to analyze.')
    data = pd.read_json(args.data, lines=True)
    print(f'Loaded {len(data)} records')
    
    data.query(f'n_water=={args.size}', inplace=True)
    print(f'Downselected to {len(data)} graphs with cluster size {args.size}\n')
    
    data['nx'] = data.apply(graph_from_dict, axis=1)
    data.reset_index(inplace=True)

    energy_dist = data['energy'].tolist()
    cluster_dist = data['nx'].tolist()

else:
    sys.exit('Data input type not supported. Supported types are "xyz" and "graph".')

# gather distributions for test data
if args.graph_properties:
    a,b,c,d,e,f,g=[],[],[],[],[],[],[]
   
    for cluster in cluster_dist:
        if args.datatype == 'xyz':
            G,_,_ = graph_loader.load_graph(cluster[2:])
        elif args.datatype == 'graph':
            G = cluster
        pG = projected_graph.project_oxygen_role_based_graph(G)
        a.append(metrics.compute_atoms(G)/3)
        b.append(len(rings.find_rings(pG, 3)))
        c.append(len(rings.find_rings(pG, 4)))
        d.append(len(rings.find_rings(pG, 5)))
        e.append(len(rings.find_rings(pG, 6)))
        f.append(metrics.compute_aspl(G))
        g.append(metrics.compute_wiener_index(G))
    
    df_dist = pd.DataFrame({'Energy': energy_dist, 'Cluster Size':a,
                            'Trimers':b, 'Tetramers':c,
                            'Pentamers':d, 'Hexamers':e,
                            'Avg Shortest Path Length':f, 'Wiener Index':g})
    features=['Energy','Trimers', 'Tetramers', 'Pentamers', 'Hexamers', 'Avg Shortest Path Length', 'Wiener Index']

else:
    df_dist = pd.DataFrame({'Energy': energy_dist})
    features = ['Energy']

# print type of statistic being calculated
if args.stat == 'kl':
    print(f"Computing KL divergence between distributions\n")
elif args.stat == 'js':
    print(f"Computing JS divergence between distributions\n") 
elif args.stat == 'ks':
    print(f"Computing KS statistic between distributions\n") 
elif args.stat == 'wd':
    print(f"Computing Wasserstein distance between distributions\n")
else:
    sys.exit("Distance computation not supported. Choose -stat kl, js, ks or wd.")

feature_table=[]
for percent in [1, 0.5, 0.1, 0.01]:
    df_comp_dist = df_comp.loc[df_comp['Cluster Size']==args.size]
    cutoff=int(len(df_comp_dist)*percent) 

    if len(df_comp_dist) > cutoff:
        cutoff_min = np.partition(df_comp_dist['Energy'].tolist(), 1)[cutoff]
    else:
        cutoff_min = np.partition(df_comp_dist['Energy'].tolist(), 1)[-1]

    df_comp_dist = df_comp_dist.loc[df_comp_dist['Energy']<=cutoff_min]


    print(f"{len(df_dist)} samples in test distribution; {len(df_comp_dist)} samples in true distribution ({int(percent*100)}% lowest energy)")

    ft=[]
    for feature in features:

        test = df_dist[feature].apply(lambda x: np.abs(x))
        true = df_comp_dist[feature].apply(lambda x: np.abs(x))
        
        if args.stat == 'kl':
            ft.append(evaluate.compute_kl_divergence(true, test, n_bins=args.bins))
        elif args.stat == 'js':
            ft.append(evaluate.compute_js_divergence(true, test, n_bins=args.bins))
        elif args.stat == 'ks':
            ft.append(evaluate.compute_ks_statistic(true, test))
        elif args.stat == 'wd':
            ft.append(evaluate.compute_wasserstein_distance(true, test))

    feature_table.append(ft)
        
feature_table=np.array(feature_table)
feature_table=[list([round(y,2) for y in x]) for x in feature_table]
feature_table=list(zip(features,feature_table[0],feature_table[1],feature_table[2], feature_table[3]))

print()
print(tabulate(feature_table, headers=['100%','50%','10%','1%'], tablefmt='psql'))
