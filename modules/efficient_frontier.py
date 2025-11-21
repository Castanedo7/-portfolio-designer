from typing import Dict, Any

import numpy as np
import pandas as pd


def simulate_efficient_frontier(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float,
    num_portfolios: int = 5000,
) -> Dict[str, Any]:
    """
    Simula portafolios aleatorios y calcula la frontera eficiente aproximada.

    Parameters
    ----------
    mean_returns : pd.Series
        Rendimientos medios diarios de los activos.
    cov_matrix : pd.DataFrame
        Matriz de covarianzas de rendimientos diarios.
    risk_free_rate : float
        Tasa libre de riesgo anual.
    num_portfolios : int
        Número de portafolios aleatorios a simular.

    Returns
    -------
    Dict[str, Any]
        frontier_df, min_risk_portfolio, max_return_portfolio, max_sharpe_portfolio
    """
    if mean_returns is None or cov_matrix is None:
        raise ValueError("mean_returns y cov_matrix no pueden ser None.")

    tickers = list(mean_returns.index)
    n_assets = len(tickers)
    trading_days = 252

    results = {
        "return": [],
        "risk": [],
        "sharpe": [],
        "weights": [],
    }

    for _ in range(num_portfolios):
        # Muestreo aleatorio de pesos (no negativos, suma=1)
        weights = np.random.dirichlet(alpha=np.ones(n_assets))

        # Anualizar
        port_return_daily = np.dot(mean_returns.values, weights)
        port_return_annual = (1 + port_return_daily) ** trading_days - 1

        port_var_daily = np.dot(weights.T, np.dot(cov_matrix.values, weights))
        port_vol_annual = np.sqrt(port_var_daily) * np.sqrt(trading_days)

        if port_vol_annual > 0:
            sharpe = (port_return_annual - risk_free_rate) / port_vol_annual
        else:
            sharpe = 0.0

        results["return"].append(port_return_annual)
        results["risk"].append(port_vol_annual)
        results["sharpe"].append(sharpe)
        results["weights"].append(weights)

    frontier_df = pd.DataFrame(
        {
            "return": results["return"],
            "risk": results["risk"],
            "sharpe": results["sharpe"],
        }
    )

    # Portafolio de mínimo riesgo
    idx_min_risk = frontier_df["risk"].idxmin()
    min_risk_portfolio = {
        "return": frontier_df.loc[idx_min_risk, "return"],
        "risk": frontier_df.loc[idx_min_risk, "risk"],
        "sharpe": frontier_df.loc[idx_min_risk, "sharpe"],
        "weights": results["weights"][idx_min_risk],
    }

    # Portafolio de máximo rendimiento
    idx_max_return = frontier_df["return"].idxmax()
    max_return_portfolio = {
        "return": frontier_df.loc[idx_max_return, "return"],
        "risk": frontier_df.loc[idx_max_return, "risk"],
        "sharpe": frontier_df.loc[idx_max_return, "sharpe"],
        "weights": results["weights"][idx_max_return],
    }

    # Portafolio de máximo Sharpe
    idx_max_sharpe = frontier_df["sharpe"].idxmax()
    max_sharpe_portfolio = {
        "return": frontier_df.loc[idx_max_sharpe, "return"],
        "risk": frontier_df.loc[idx_max_sharpe, "risk"],
        "sharpe": frontier_df.loc[idx_max_sharpe, "sharpe"],
        "weights": results["weights"][idx_max_sharpe],
    }

    return {
        "frontier_df": frontier_df,
        "min_risk_portfolio": min_risk_portfolio,
        "max_return_portfolio": max_return_portfolio,
        "max_sharpe_portfolio": max_sharpe_portfolio,
    }
