import copy
import math

import torch
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.utils import add_self_loops

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def neighborhood_difficulty_measurer(data, label):
    # 加上自环，将节点本身的标签也计算在内
    smi = data.x / torch.norm(data.x, dim=-1, keepdim=True)  # 方差归一化，即除以各自的模
    smi = torch.mm(smi, smi.T)  # 矩阵乘法
    # smi = 1 - smi * data.A  # 乘以邻接矩阵
    smi = smi * data.A  # 乘以邻接矩阵
    local_difficulty = torch.sum(smi, dim=-1) / torch.sum(data.A, dim=-1)  # 按行求和
    local_difficulty = 1 - local_difficulty / local_difficulty.sum()
    return local_difficulty.to(device)


def feature_difficulty_measurer(data, label, embedding):
    # global_difficulty = 0.0
    importances = torch.zeros(data.A.shape[1])
    A = copy.deepcopy(data.A)
    A = torch.diag(torch.ones(A.shape[0]), diagonal=0)
    D = torch.sum(A, dim=0) + torch.sum(A, dim=1) - A.diagonal()
    IG = torch.log(torch.sum(D, dim=0)) - torch.sum(D / D.sum(dim=0) * torch.log(D), dim=0)

    for i in range(A.shape[0]):
        A_modified = copy.deepcopy(A)

        A_modified = A_modified[:, torch.cat((torch.arange(0, i), torch.arange(i + 1, A.shape[0])), dim=0)]
        A_modified = A_modified[torch.cat((torch.arange(0, i), torch.arange(i + 1, A.shape[0])), dim=0), :]
        # print(A_modified)/

        D_modified = A_modified.sum(axis=0) + A_modified.sum(axis=1) - A_modified.diagonal()

        IG_modified = torch.log(torch.sum(D_modified, dim=0)) - torch.sum(
            D_modified / D_modified.sum(dim=0) * torch.log(D_modified), dim=0)

        im = IG - IG_modified
        importances[i] = im
    importances = 1 - importances / importances.sum()
    return importances.to(device)

# multi-perspective difficulty measurer
def difficulty_measurer(data, label, embedding, alpha):
    local_difficulty = neighborhood_difficulty_measurer(data, label)
    global_difficulty = feature_difficulty_measurer(data, label, embedding)
    # print('local_difficulty.shape', local_difficulty.shape)
    # print('global_difficulty.shape', global_difficulty.shape)
    node_difficulty = alpha * local_difficulty + (1 - alpha) * global_difficulty
    return node_difficulty


# sort training nodes by difficulty
def sort_training_nodes(data, label, embedding, alpha=0.5):
    node_difficulty = difficulty_measurer(data, label, embedding, alpha)
    _, indices = torch.sort(node_difficulty)
    sorted_trainset = data.train_id[indices]
    return sorted_trainset


def training_scheduler(lam, t, T, scheduler='linear'):
    if scheduler == 'linear':
        return min(1, lam + (1 - lam) * t / T)
    elif scheduler == 'root':
        return min(1, math.sqrt(lam ** 2 + (1 - lam ** 2) * t / T))
    elif scheduler == 'geom':
        return min(1, 2 ** (math.log2(lam) - math.log2(lam) * t / T))
