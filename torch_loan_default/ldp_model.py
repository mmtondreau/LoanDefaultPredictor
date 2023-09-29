# Import required packages

# Data packages

#


# Visualization Packages

# Import any other packages you may want to use
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl

# PyTorch TensorBoard support

# from lightning.pytorch.loggers import NeptuneLogger


class Block(pl.LightningModule):
    def __init__(self, input_size, hidden_units, dropout=0.2, activation=F.relu):
        super(Block, self).__init__()
        self.layer = nn.Linear(input_size, hidden_units)
        self.drop = nn.Dropout(dropout)
        self.batchNorm = nn.BatchNorm1d(hidden_units)
        self.activation = activation

    def forward(self, x):
        x = self.layer(x)
        x = self.batchNorm(x)
        x = self.drop(x)
        x = self.activation(x)
        return x


class LDPModel(pl.LightningModule):
    def __init__(self, num_features, num_classes, hidden_units):
        super(LDPModel, self).__init__()
        all_layers = []
        for hidden_unit in hidden_units:
            all_layers.append(Block(input_size=num_features, hidden_units=hidden_unit))
            num_features = hidden_unit
        all_layers.append(nn.Linear(hidden_units[-1], num_classes))
        self.layers = nn.Sequential(*all_layers)

    def forward(self, x):
        x = self.layers(x)
        return F.sigmoid(x)
