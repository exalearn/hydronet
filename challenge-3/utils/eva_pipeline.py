import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from analysis_scripts import graph_loader, projected_graph, rings, metrics, evaluate, calculate_rmsd
from analysis_scripts.data import graph_from_dict
import tempfile
import sys
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import os.path as op

base_path = os.getcwd()

def load_data(data_path, data_type, cluster_size):
    """
    Extract energy and cluster size of loaded data

    Argument:
    datatype (str) -- specify data type either xyz or graph
    data_path (str) -- file path
    cluster_size (int) -- the number of cluster

    Output:
    energy_dist (list) -- list contain energy profile
    cluster_dist (list) -- list contain structural profile
    """
    if data_type == 'xyz':
        cluster_dist, energy_dist = graph_loader.read_lines_base(data_path)
        # check for only one cluster size in test data
        if cluster_size*3 != np.array([len(x)-2 for x in cluster_dist]).mean():
            sys.exit('Multiple cluster sizes found in test data. Please split test data into different files based on cluster size.')

    elif data_type == 'graph':
        if cluster_size == 0:
            sys.exit('Use -size flag to choose the cluster size to analyze.')
        data = pd.read_json(data_path, lines=True)
        print(f'Loaded {len(data)} records')

        data.query(f'n_water=={cluster_size}', inplace=True)
        print(f'Downselected to {len(data)} graphs with cluster size {cluster_size}\n')

        data['nx'] = data.apply(graph_from_dict, axis=1)
        data.reset_index(inplace=True)

        energy_dist = data['energy'].tolist()
        cluster_dist = data['nx'].tolist()

    return energy_dist, cluster_dist

def profile_creation(data_type, cluster, energy):
    """
    This function used to calculate the cycle distribution, e.g. how many hexagone, pentagone within the molecule

    Argument:
    datatype (str) -- specify data either xyz or graph
    cluster (list) -- structural profile
    energy (list)  -- energetic profile

    Output:
    df_dist (dataframe) -- containing energy and cycle profiles
    """
    a,b,c,d,e,f,g = [],[],[],[],[],[],[]
    for cluster in cluster:
        if data_type == 'xyz':
            G,_,_ = graph_loader.load_graph(cluster[2:])
        elif data_type == 'graph':
            G = cluster
        pG = projected_graph.project_oxygen_role_based_graph(G)
        a.append(metrics.compute_atoms(G)/3)
        b.append(len(rings.find_rings(pG, 3)))
        c.append(len(rings.find_rings(pG, 4)))
        d.append(len(rings.find_rings(pG, 5)))
        e.append(len(rings.find_rings(pG, 6)))
        f.append(metrics.compute_aspl(G))
        g.append(metrics.compute_wiener_index(G))

    df_dist = pd.DataFrame({'Energy': energy, 'Cluster Size':a,
                            'Trimers':b, 'Tetramers':c,
                            'Pentamers':d, 'Hexamers':e,
                            'Avg Shortest Path Length':f, 'Wiener Index':g})
    return df_dist

def statistical_evaluation(df_test, stat_type, cluster_size, bins=100):
    """
    Compare the statistics difference between test and ground truth data.

    Argument:
    df_test (dataframe) -- test dataframe containing energy and cycle profiles
    stat_type (str) -- specify type of statistical analysis, kl, js, ks, and wd
    cluster_size (int) -- size of cluster

    Output:
    df_ft (dataframe) -- dataframe contain statistical analysis
    """
    features=['Energy','Trimers', 'Tetramers', 'Pentamers', 'Hexamers',
              'Avg Shortest Path Length', 'Wiener Index','rmsd','Graph Similarity','Projected Similarity']
    df_comp = pd.read_csv(op.join(base_path,'data/output/feature_1kcal.csv'))
    df_comp_dist = df_comp.loc[df_comp['Cluster Size']==cluster_size]

    feature_table=[]
    ft=[]
    for feature in features:

        test = df_test[feature].apply(lambda x: np.abs(x))
        true = df_comp_dist[feature].apply(lambda x: np.abs(x))

        if stat_type == 'kl':
            ft.append(evaluate.compute_kl_divergence(true, test, n_bins=bins))
        elif stat_type == 'js':
            ft.append(evaluate.compute_js_divergence(true, test, n_bins=bins))
        elif stat_type == 'ks':
            ft.append(evaluate.compute_ks_statistic(true, test))
        elif stat_type == 'wd':
            ft.append(evaluate.compute_wasserstein_distance(true, test))
    df_ft = pd.DataFrame(columns=features)
    df_ft.loc[0]=ft
    return df_ft

def coord_similarity(cluster_dist, cluster_size):
    """
    Compare the structural similarity of each input coordinates with the structure that has
    the lowest energy of same cluster size

    Argument:
    cluster_dist (list) -- list containing structural profile of test set
    cluster_size (int) -- size of cluster

    Output:
    df_rmsd (dataframe) -- dataframe including geometrical similarity
    """
    test_coord = []
    for cluster in cluster_dist:
        tmp = tempfile.NamedTemporaryFile()

        # Open the file for writing.
        with open(tmp.name, 'w') as f:
            for item in cluster:
                if type(item) == int or type(item) == float:
                    f.write(str(item))
                    f.write("\n")
                else:
                    f.write(str(item))

        coords = calculate_rmsd.get_coordinates(tmp.name,'xyz')[1]
        test_coord.append(coords)
    comp_coord = calculate_rmsd.get_coordinates(op.join(base_path,'data/lowest_xyz/w{}_lowest.xyz'.format(cluster_size)), 'xyz')[1]
    rmsd = []
    for item in test_coord:
        similarity = calculate_rmsd.kabsch_weighted_rmsd(item, comp_coord)
        rmsd.append(similarity)
    df_rmsd = pd.DataFrame(rmsd, columns= ['rmsd'])
    return df_rmsd

def graph_similarity(cluster_list):

    similarity_list, projected_similarity_list=[],[]
    for i, cluster in enumerate(cluster_list):
        cluster_size = int(int(len(cluster))/3)
        xyzfile= f'w{cluster_size}_lowest.xyz'
        base_cluster_list,_ = graph_loader.read_lines_base(op.join(base_path, 'data/lowest_xyz/',xyzfile))
        #load lowest energy structure from the full list
        G_base,_,_ = graph_loader.load_graph(base_cluster_list[0][2:])
        proj_G_base = projected_graph.project_oxygen_role_based_graph(G_base)

        try:
            #Similarity of full graph
            G_variable,_,_ = graph_loader.load_graph(cluster[2:])
            similarity_list.append(metrics.calculate_similarity(G_base, G_variable))
            #Similarity of oxygen projected graph
            proj_G_variable = projected_graph.project_oxygen_role_based_graph(G_variable)
            projected_similarity_list.append(metrics.calculate_similarity(proj_G_base, proj_G_variable))
        except:
            #disconnected graphs fail test and get s values of -1
            projected_similarity_list.append(-1)
            similarity_list.append(-1)
    d = {'Graph Similarity': similarity_list,'Projected Similarity': projected_similarity_list}
    df_grap_sim = pd.DataFrame(d)
    return df_grap_sim

def main(data_path, data_type, cluster_size, stat_type):

    # load test data, create two lists including energetic and structural profile
    ene_prof, stru_prof = load_data(data_path, data_type, cluster_size)
    # create cycyle information from structural profile
    cyc_prof = profile_creation(data_type, stru_prof, ene_prof)
    # evaluate geometrical similarity between test set and lowest structure
    geom_similarity = coord_similarity(stru_prof, cluster_size)
    gra_similarity = graph_similarity(stru_prof)
    df_summary = pd.concat([cyc_prof, geom_similarity,gra_similarity], axis=1)
    # analyze statistical difference between test set and ground truth
    stat_eva = statistical_evaluation(df_summary, stat_type, cluster_size)

    return stat_eva, df_summary
