# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/models/model.dlinear_robust.ipynb.

# %% auto 0
__all__ = ['device', 'SeriesDecompose']

# %% ../../nbs/models/model.dlinear_robust.ipynb 1
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn

# %% ../../nbs/models/model.dlinear_robust.ipynb 2
device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

# %% ../../nbs/models/model.dlinear_robust.ipynb 4
import torch.nn.functional as F
from statsmodels.tsa.seasonal import seasonal_decompose


class SeriesDecompose(nn.Module):
    def __init__(self, kernel_size, period=12):
        """
        Decomposes a time series into trend, seasonal, and residual components.

        :param kernel_size: The size of the kernel for moving average.
        :param period: The seasonal period for decomposition.
        """
        super(SeriesDecompose, self).__init__()
        self.kernel_size = kernel_size
        self.period = period
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):
        device = x.device  # Ensure consistency with input tensor device

        if x.dim() == 3:  # If input is 3D (batch, features, seq_len)
            batch, features, seq_len = x.shape
            x = x.view(batch, features * seq_len)  # Flatten only across sequence

        # Convert to NumPy (only first batch)
        x_np = x[0].detach().cpu().numpy()  # Convert only the first time series

        # Perform seasonal decomposition
        decompose = seasonal_decompose(
            x_np, period=self.period, model="additive", extrapolate_trend="freq"
        )

        # Convert back to PyTorch tensors
        trend = torch.tensor(decompose.trend, device=device, dtype=x.dtype)
        seasonal = torch.tensor(decompose.seasonal, device=device, dtype=x.dtype)
        residual = torch.tensor(decompose.resid, device=device, dtype=x.dtype)

        return trend, seasonal, residual


# class SeriesDecompose(nn.Module):
#     def __init__(self,kernel_size):
#         super(SeriesDecompose, self).__init__()
#         self.kernel_size = kernel_size
#         self.avg = nn.AvgPool1d(kernel_size=kernel_size,stride=1, padding=0)

#     def forward(self,x):
#         if x.dim() == 3:  # If input is 3D (batch, features, seq_len)
#             batch, features, seq_len = x.shape
#             x = x.view(batch ,features *  seq_len)

#         # apply padding to keep the result to be of same size
#         x_padded = torch.cat([x[:,0:1].repeat(1, (self.kernel_size - 1 )//2),
#              x,
#              x[:,-1:].repeat(1,(self.kernel_size - 1 )//2)],dim=1)
#         # moving_avg = self.avg(x_padded)  #.squeeze(1)
#         decompose = seasonal_decompose(x.flatten().cpu(),period=12)

#         # residual = x - moving_avg
#         return torch.tensor(decompose.trend,device=device) , torch.tensor(decompose.seasonal,device=device) , torch.tensor(decompose.resid,device=device)
