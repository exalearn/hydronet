import numpy as np
import pandas as pd
import networkx as nx

def generate_with_constraints(n_waters, bidirectional=False):
    '''
    constraints: 
        edges: max 2 in; max 2 out; no self loops; no edges go in both directions
    '''
    
    # set random in and out degrees, 2 max for each
    in_degrees = np.random.randint(0, high=3, size=n_waters)
    out_degrees = np.random.randint(0, high=3, size=n_waters)

    # check for and fix disconnected water molecules
    for n in range(n_waters):
        while (in_degrees[n]==0) and (out_degrees[n]==0):
            in_degrees[n]=np.random.randint(0, high=3, size=1)
            out_degrees[n]=np.random.randint(0, high=3, size=1)
            #print(f'reconnected water {n}')


    # sum of in and out degrees must be equal
    # make list same length by adding to lowest to get highest
    if in_degrees.sum() != out_degrees.sum():
        to_add = np.abs(in_degrees.sum() - out_degrees.sum())

        if in_degrees.sum() < out_degrees.sum():
            # add to random spots to 
            # get indices < 2
            idxes = np.concatenate((np.where(in_degrees<2)[0],np.where(in_degrees<1)[0]))
            for i in np.random.choice(idxes, size=to_add, replace=False):
                in_degrees[i]+=1
        elif in_degrees.sum() > out_degrees.sum():
            idxes = np.concatenate((np.where(out_degrees<2)[0],np.where(out_degrees<1)[0]))
            for i in np.random.choice(idxes, size=to_add, replace=False):
                out_degrees[i]+=1

    G = nx.directed_configuration_model(in_degrees, out_degrees, create_using=nx.DiGraph)

    # no self loops
    G.remove_edges_from(nx.selfloop_edges(G))
    
    # remove edges that go in both directions
    for edge in G.to_undirected(reciprocal=True).edges:
        edge = list(edge)
        np.random.shuffle(edge)
        G.remove_edge(edge[0],edge[1])
        #print(f'removed edge {edge[0],edge[1]}')

    # add node and edge attributes
    nx.set_node_attributes(G, 'O', "label")
    nx.set_edge_attributes(G, 'donate', "label")
        
    # for input into MPNN
    if bidirectional:
        # add accept edges in the reverse direction
        G.add_edges_from(np.array(G.edges)[:,::-1], label='accept')

    return G


def generate_random_graph(n_waters, bidirectional=False):
    G =  generate_with_constraints(n_waters, bidirectional=bidirectional)
    
    # redo if graph isnt connected
    while nx.number_connected_components(G.to_undirected())>1:
        G =  generate_random_graph(n_waters)
        #print('disconnected - regenerating')

    G=set_category(G)
    return G


def generate_random_sample(n_waters, n=1000):
    labels=['trimers','tetramers','pentamers','hexamers','shortest_path','wiener']
    data=np.vstack([metrics.compute_metrics(generate_random_graph(n_waters)) for i in range(n)]).T

    df=pd.DataFrame({labels[i]:data[i] for i in range(len(labels))})

    return df


def set_category(G):
    [nx.set_edge_attributes(G, {edge: {"cat":1}}) if G.edges[edge]['label']=='donate' else nx.set_edge_attributes(G, {edge: {"cat":0}}) for edge in G.edges]
    return G


