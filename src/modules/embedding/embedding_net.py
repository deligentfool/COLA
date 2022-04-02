import torch
import torch.nn.functional as F
import torch.nn as nn


class Embedding_net(nn.Module):
    def __init__(self, args):
        super(Embedding_net, self).__init__()
        self.args = args
        self.embedding_layer = nn.Embedding(self.args.perceive_dim + 1, args.perceive_embedding_dim)

    def forward(self, id):
        return self.embedding_layer(id)
