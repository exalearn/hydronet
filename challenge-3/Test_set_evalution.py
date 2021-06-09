import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from analysis_scripts import graph_loader, projected_graph, rings, metrics, evaluate, calculate_rmsd, eva_pipeline
from analysis_scripts.data import graph_from_dict
import matplotlib.pyplot as plt
import tempfile
import sys
import seaborn as sns
import os.path as op
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-data', required=True, type=str, help="Test data.")
parser.add_argument('-gt', default='/data/output/feature_1kcal.csv', type=str, help="Ground truth data for comparison.")
parser.add_argument('--N', default=12, type=int, help="cluster size")
parser.add_argument('--stat', default='wd', type=str, help="stat to use")
args = parser.parse_args()


# Load test set data
stat_info, summary = eva_pipeline.main(args.data,'xyz', args.N, args.stat)
cwd = os.getcwd()
plot_output = op.join(cwd, 'Plots')

if not op.isdir(plot_output):
    os.mkdir(plot_output)

lowest_energy = eva_pipeline.load_data('data/lowest_xyz/w{}_lowest.xyz'.format(args.N),'xyz',args.N)[0][0]

# Energy disrtibution
sns.set_style("darkgrid")
sns_plot = sns.displot(summary, x="Energy",kind ='kde',rug=True,
                       fill=True,aspect=11.7/8.27,palette=['red'], linewidth=2.5)
plt.axvline(lowest_energy, 0,1,color='black')
plt.savefig(op.join(plot_output,'energy_distribution.png'),dpi=150)
plt.close()


# Graph similarity distribution
sns_plot = sns.displot(summary, x="Graph Similarity",kind ='kde',rug=True,
                       fill=True,aspect=11.7/8.27,palette=['red'], linewidth=2.5)
plt.savefig(op.join(plot_output,'graph_similarity.png'),dpi=150)
plt.close()



# Projected distribution
sns_plot = sns.displot(summary, x="Projected Similarity",rug=True,
                       kind ='kde',fill=True,aspect=11.7/8.27,palette=['red'], linewidth=2.5)
plt.savefig(op.join(plot_output,'projected_graph_similarity_distribution.png'),dpi=150)
plt.close()


# similarities on one plot
sns_plot = sns.displot(summary, x="Graph Similarity", y="Projected Similarity",
                       kind ='kde')
plt.savefig(op.join(plot_output,'both_graph_similarity_distributions.png'),dpi=150)
plt.close()
