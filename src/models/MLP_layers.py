import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiLayerPerceptron(torch.nn.Module):
    """
    Class to instantiate a Multilayer Perceptron model
    """

    def __init__(self, input_dim, embed_dims, dropout, output_layer=False):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        y = self.mlp(x)
        return y