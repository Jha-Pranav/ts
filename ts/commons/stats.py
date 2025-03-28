# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/commons/commons.stats.ipynb.

# %% auto 0
__all__ = ['ensure_tensor', 'get_seasonality', 'extract_stats_features']

# %% ../../nbs/commons/commons.stats.ipynb 1
from itertools import groupby

import antropy as ant
import numpy as np
import pandas as pd
import pywt
import scipy.stats as stats
import statsmodels.api as sm
import torch
from nolds import dfa, hurst_rs
from scipy.fftpack import fft
from scipy.signal import find_peaks, welch
from scipy.stats import variation
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import STL, seasonal_decompose
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

# %% ../../nbs/commons/commons.stats.ipynb 2
def ensure_tensor(series):
    if isinstance(series, pd.Series):
        return torch.tensor(series.values, dtype=torch.float32)
    elif isinstance(series, np.ndarray):
        return torch.tensor(series, dtype=torch.float32)
    elif isinstance(series, torch.Tensor):
        return series.float()
    else:
        raise ValueError("Input must be a pandas Series, NumPy array, or PyTorch tensor")


def get_seasonality(series_name):
    mapping = {"D": 7, "W": 4, "M": 12, "Q": 4, "Y": 2}
    if not series_name:
        return 6
    return mapping.get(series_name[0], 6)  # Default period is 6 if not found


def extract_stats_features(series, max_lag=10):
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]  # Take first column if DataFrame
    if series.empty:
        return np.nan

    name, series = series.name, torch.tensor([i for i in series.values if i])
    series = ensure_tensor(series)
    series_np = series.cpu().numpy()
    features = {}

    # Basic Stats
    features["mean"] = torch.mean(series).item()
    features["std"] = torch.std(series).item()
    features["var"] = torch.var(series).item()
    features["skewness"] = stats.skew(series_np)
    features["kurtosis"] = stats.kurtosis(series_np)
    features["min"] = torch.min(series).item()
    features["max"] = torch.max(series).item()
    features["range"] = features["max"] - features["min"]
    features["median"] = torch.median(series).item()
    features["iqr"] = np.percentile(series_np, 75) - np.percentile(series_np, 25)
    features["mad"] = torch.mean(torch.abs(series - torch.mean(series))).item()
    features["medad"] = torch.median(torch.abs(series - torch.median(series))).item()
    features["cv"] = variation(series_np)

    # Normality Tests
    _, features["shapiro_p"] = stats.shapiro(series_np)
    _, features["ks_p"] = stats.kstest(series_np, "norm")
    _, features["jarque_bera_p"] = stats.jarque_bera(series_np)

    # Stationarity Tests
    features["adf_p"] = adfuller(series_np)[1]
    features["kpss_p"] = kpss(series_np, regression="c")[1]

    # Autocorrelation
    max_possible_lag = len(series_np) // 2
    max_lag = min(max_lag, max_possible_lag)
    acf_values = acf(series_np, nlags=max_lag)
    pacf_values = pacf(series_np, nlags=max_lag)
    for lag in range(1, max_lag + 1):
        features[f"acf_{lag}"] = abs(acf_values[lag])
        features[f"pacf_{lag}"] = abs(pacf_values[lag])

    # Seasonal Strength & Trend Features
    seasonality_period = get_seasonality(name)
    if len(series_np) >= 2 * seasonality_period:
        stl = STL(series_np, period=seasonality_period).fit()
        features["stl_trend_std"] = np.std(stl.trend)
        features["stl_seasonal_std"] = np.std(stl.seasonal)
        features["stl_resid_std"] = np.std(stl.resid)
        features["seasonal_strength"] = 1 - (np.var(stl.resid) / np.var(stl.seasonal))

    x = np.arange(len(series_np))
    slope, intercept = np.polyfit(x, series_np, 1)
    features["trend_slope"] = slope
    features["trend_curvature"] = np.polyfit(x, series_np, 2)[0]

    # Recurrence & Complexity Measures
    features["recurrence_rate"] = np.sum(np.diff(series_np) == 0) / len(series_np)
    features["determinism"] = np.sum(np.diff(series_np) > 0) / len(series_np)
    # features["lz_complexity"] = ant.lziv_complexity(series_np, normalize=True)
    # features["corr_dimension"] = ant.corr_dim(series_np, emb_dim=2)

    # Nonlinearity
    features["time_reversibility"] = np.mean((series_np[:-1] - series_np[1:]) ** 3)

    # Longest flat segment
    features["longest_flat_segment"] = max(
        [len(list(g)) for k, g in groupby(series_np) if k == 0], default=0
    )

    return pd.Series(features)
