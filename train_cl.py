import argparse

import tensorflow as tf
from numpy.random import seed
from sklearn import metrics

from utils.SingleCell import SingleCell
from utils.early_stop import EarlyStop
from utils.preprocess import *

seed(1)
tf.random.set_seed(1)

# Remove warnings
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from sccheb_cl import SCTAG
from utils.graph_function import *
import os
import wandb


# Compute cluster centroids, which is the mean of all points in one cluster.
def computeCentroids(data, labels):
    n_clusters = len(np.unique(labels))
    return np.array([data[labels == i].mean(0) for i in range(n_clusters)])


def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment as linear_assignment
    ind = linear_assignment(w.max() - w)
    ind = np.array((ind[0], ind[1])).T
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--expname", default="exp", type=str)
    parser.add_argument("--dataname", default="Quake_Smart-seq2_Limb_Muscle", type=str)
    parser.add_argument("--highly_genes", default=500, type=int)
    parser.add_argument("--pretrain_epochs", default=1000, type=int)
    parser.add_argument("--maxiter", default=500, type=int)
    parser.add_argument("--gpu_option", default="0")
    parser.add_argument("--prune_epoch", default=0.0, type=int)
    parser.add_argument("--k", default=20, type=int)
    parser.add_argument("--is_datadrop", action='store_true', default=False)
    parser.add_argument("--sample_method", default='cl_sample', type=str)
    parser.add_argument("--latent_dim", default=15, type=int)
    parser.add_argument("--is_cl", action='store_true')
    args = parser.parse_args()

    wandb.login()
    wandb.init(
        # Set the project where this run will be logged
        project="scCLG_CL",
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name=f"{args.dataname}_gpu_epoch_{args.prune_epoch}",
        # Track hyperparameters and run metadata
        config={
            "dataset": args.dataname,
            "highly_genes": args.highly_genes,
            "pretrain_epochs": args.pretrain_epochs,
            "fit_epochs": args.maxiter,
            "prune_epoch": args.prune_epoch,
            "k": args.k,
            "sample_method": args.sample_method,
            "latent_dim": args.latent_dim
        })

    # Load data
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_option
    x, y = prepro('./data/' + args.dataname + '/data.h5')

    # x = np.ceil(x).astype(np.int)
    x = np.ceil(x).astype('int')
    cluster_number = int(max(y) - min(y) + 1)
    adata = sc.AnnData(x)
    adata.obs['Group'] = y
    adata = normalize(adata, copy=True, highly_genes=args.highly_genes, size_factors=True, normalize_input=True,
                      logtrans_input=True)
    count = adata.X

    sorted_trainset = np.loadtxt(f'data_dropouted/{args.dataname}_sorted_trainset_{args.highly_genes}_gpu.txt', dtype=int)

    adj, adj_n = get_adj(count, k=args.k)
    # adj matrix-> adj list
    edge = None
    ed = sp.coo_matrix(adj)
    indices = np.vstack((ed.row, ed.col))
    edge_index = indices
    data = SingleCell(count, edge_index, adj)
    # Pre-training
    # Build model
    model = SCTAG(count, adj=adj, adj_n=adj_n, latent_dim=args.latent_dim, wandb=wandb, prune_epoch=args.prune_epoch)

    model.pre_train(epochs=args.pretrain_epochs)

    Y = model.embedding(count, adj_n)
    from sklearn.cluster import SpectralClustering
    labels = SpectralClustering(n_clusters=cluster_number, affinity="precomputed", assign_labels="discretize",
                                random_state=0).fit_predict(adj)
    centers = computeCentroids(Y, labels)

    patience = 100
    early_stop1 = EarlyStop(patience)

    if args.is_cl:
        Cluster_predicted = model.alt_train_cl(y, epochs=args.maxiter, centers=centers, sorted_trainset=sorted_trainset, early_stop=early_stop1)
    else:
        Cluster_predicted = model.alt_train(y, epochs=args.maxiter, centers=centers, early_stop=early_stop1)

    if y is not None:
        acc = np.round(cluster_acc(y, Cluster_predicted.y_pred), 5)
        y = list(map(int, y))
        Cluster_predicted.y_pred = np.array(Cluster_predicted.y_pred)
        nmi = np.round(metrics.normalized_mutual_info_score(y, Cluster_predicted.y_pred), 5)
        ari = np.round(metrics.adjusted_rand_score(y, Cluster_predicted.y_pred), 5)
        print('ACC= %.4f, ARI= %.4f, NMI= %.4f' % (acc, ari, nmi))

    wandb.finish()
