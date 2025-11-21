from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats


def compute_returns(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula rendimientos diarios (simple percentage change).

    Parameters
    ----------
    prices_df : pd.DataFrame
        DataFrame de precios ajustados.

    Returns
    -------
    pd.DataFrame
        DataFrame de rendimientos diarios.
    """
    return prices_df.pct_change().dropna(how="all")


def get_portfolio_timeseries(
    returns_df: pd.DataFrame,
    weights: np.ndarray,
) -> pd.Series:
    """
    Calcula la serie de rendimientos del portafolio.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Rendimientos diarios de cada activo.
    weights : np.ndarray
        Pesos del portafolio (suma = 1.0).

    Returns
    -------
    pd.Series
        Rendimientos diarios del portafolio.
    """
    weights = np.array(weights)
    weights = weights / weights.sum()
    portfolio_returns = returns_df.dot(weights)
    portfolio_returns.name = "Portfolio"
    return portfolio_returns


def compute_basic_stats(
    portfolio_returns: pd.Series,
    risk_free_rate: float,
) -> Dict[str, float]:
    """
    Calcula estadísticas básicas anualizadas del portafolio.

    Parameters
    ----------
    portfolio_returns : pd.Series
        Rendimientos diarios del portafolio.
    risk_free_rate : float
        Tasa libre de riesgo anual (decimales).

    Returns
    -------
    Dict[str, float]
        annual_return, annual_volatility, sharpe_ratio, skewness, max_drawdown
    """
    if portfolio_returns is None or portfolio_returns.empty:
        return {
            "annual_return": 0.0,
            "annual_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "skewness": 0.0,
            "max_drawdown": 0.0,
        }

    mean_daily = portfolio_returns.mean()
    std_daily = portfolio_returns.std()

    trading_days = 252
    annual_return = (1 + mean_daily) ** trading_days - 1
    annual_volatility = std_daily * np.sqrt(trading_days)

    if annual_volatility > 0:
        sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
    else:
        sharpe_ratio = 0.0

    skewness = float(stats.skew(portfolio_returns.values, bias=False))

    cumulative = (1 + portfolio_returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative / running_max) - 1
    max_drawdown = drawdown.min()

    return {
        "annual_return": float(annual_return),
        "annual_volatility": float(annual_volatility),
        "sharpe_ratio": float(sharpe_ratio),
        "skewness": skewness,
        "max_drawdown": float(max_drawdown),
    }


def compute_beta(
    portfolio_returns: pd.Series,
    benchmark_returns: Optional[pd.DataFrame],
) -> Optional[float]:
    """
    Calcula la beta del portafolio respecto a un benchmark.

    Parameters
    ----------
    portfolio_returns : pd.Series
        Rendimientos diarios del portafolio.
    benchmark_returns : Optional[pd.DataFrame]
        DataFrame con rendimientos del benchmark (una columna).

    Returns
    -------
    Optional[float]
        Beta del portafolio. None si no hay datos suficientes.
    """
    if (
        portfolio_returns is None
        or portfolio_returns.empty
        or benchmark_returns is None
        or benchmark_returns.empty
    ):
        return None

    # Alinear por fechas
    combined = pd.concat([portfolio_returns, benchmark_returns], axis=1, join="inner").dropna()
    if combined.shape[0] < 5:
        return None

    port = combined.iloc[:, 0]
    bench = combined.iloc[:, 1]

    cov = np.cov(port, bench)[0, 1]
    var_bench = np.var(bench)
    if var_bench == 0:
        return None

    beta = cov / var_bench
    return float(beta)
