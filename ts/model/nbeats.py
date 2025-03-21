# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/models/model.nbeats.ipynb.

# %% auto 0
__all__ = ['device']

# %% ../../nbs/models/model.nbeats.ipynb 1
import pytorch_lightning as pl
import torch
import torch.nn as nn

# %% ../../nbs/models/model.nbeats.ipynb 2
device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
