import os

import tensorflow as tf
import torch
from numpy.random import seed

from SingleCell import SingleCell
from preprocess import *
from util_clnode import sort_training_nodes

seed(1)
tf.random.set_seed(1)

# Remove warnings
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from sccheb_cl import SCTAG
from graph_function import *


# Compute cluster centroids, which is the mean of all points in one cluster.
def computeCentroids(data, labels):
    n_clusters = len(np.unique(labels))
    return np.array([data[labels == i].mean(0) for i in range(n_clusters)])


os.environ["CUDA_VISIBLE_DEVICES"] = '0'
print(tf.config.list_physical_devices('GPU'))
# Load data
dataname = 'Wang_Lung'
highly_genes = 1000
pretrain_epochs = 1000
x, y = prepro('./data/' + dataname + '/data.h5')

x = np.ceil(x).astype('int')
cluster_number = int(max(y) - min(y) + 1)
adata = sc.AnnData(x)
adata.obs['Group'] = y
adata = normalize(adata, copy=True, highly_genes=highly_genes, size_factors=True, normalize_input=True,
                  logtrans_input=True)
count = adata.X
count_tmp = adata.X

# Build model
adj, adj_n = get_adj(count, k=20)

# adj matrix-> adj list
edge = None
ed = sp.coo_matrix(adj)
indices = np.vstack((ed.row, ed.col))
edge_index = indices

data = SingleCell(adata.X, edge_index, adj)

# Pre-training and remove hard sample
model1 = SCTAG(count, adj=adj, adj_n=adj_n)

model1.pre_train(epochs=pretrain_epochs)

Y = model1.embedding(count, adj_n)

labels = None

sorted_trainset = sort_training_nodes(data, labels, torch.tensor(Y), alpha=0.5)
sorted_trainset_ = sorted_trainset.cpu().numpy()
np.savetxt(f'data_dropouted/{dataname}_sorted_trainset_{highly_genes}_gpu.txt', sorted_trainset_, fmt='%d')

