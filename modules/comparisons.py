from typing import List, Optional

import pandas as pd
import yfinance as yf


def build_comparison_df(
    portfolio_returns: pd.Series,
    benchmark_tickers: List[str],
) -> Optional[pd.DataFrame]:
    """
    Construye un DataFrame de rendimientos acumulados (índice 100)
    para el portafolio y una lista de benchmarks.

    Parameters
    ----------
    portfolio_returns : pd.Series
        Serie de rendimientos diarios del portafolio.
    benchmark_tickers : List[str]
        Lista de tickers benchmark (QQQ, SPY, GLD, etc.).

    Returns
    -------
    Optional[pd.DataFrame]
        DataFrame con columnas: 'Portfolio', 'QQQ', 'SPY', etc.
    """
    if portfolio_returns is None or portfolio_returns.empty:
        return None

    start_date = portfolio_returns.index.min()
    end_date = portfolio_returns.index.max()

    # Rendimiento acumulado del portafolio (normalizado a 100)
    cum_port = (1 + portfolio_returns).cumprod()
    cum_port = cum_port / cum_port.iloc[0] * 100.0
    df = pd.DataFrame({"Portfolio": cum_port})

    for t in benchmark_tickers:
        try:
            data = yf.Ticker(t).history(start=start_date, end=end_date, auto_adjust=True)
            if data is None or data.empty:
                continue
            bench_returns = data["Close"].pct_change().dropna()
            # Alinear con fechas del portafolio
            combined = pd.concat([portfolio_returns, bench_returns], axis=1, join="inner")
            bench_returns_aligned = combined.iloc[:, 1]
            cum_bench = (1 + bench_returns_aligned).cumprod()
            cum_bench = cum_bench / cum_bench.iloc[0] * 100.0
            df[t] = cum_bench
        except Exception:
            continue

    df = df.dropna(how="all")
    if df.empty:
        return None

    return df
