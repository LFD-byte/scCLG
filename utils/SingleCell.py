import torch


class SingleCell:
    def __init__(self, X, edge_index, A):
        self.x = torch.DoubleTensor(X).to(torch.float32)
        self.edge_index = torch.tensor(edge_index)
        self.A = torch.DoubleTensor(A).to(torch.float32)
        self.train_id = torch.arange(0, X.shape[0]).to(torch.long)
