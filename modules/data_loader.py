from typing import List, Optional

import pandas as pd
import yfinance as yf


def get_price_data(tickers: List[str], period: str = "3y") -> Optional[pd.DataFrame]:
    """
    Descarga precios ajustados diarios para una lista de tickers usando yfinance.

    Parameters
    ----------
    tickers : List[str]
        Lista de símbolos de acciones/ETFs.
    period : str
        Periodo, por ejemplo '1y', '3y', '10y'.

    Returns
    -------
    Optional[pd.DataFrame]
        DataFrame con precios de cierre ajustados. Índice = fechas.
    """
    if not tickers:
        return None

    prices_dict = {}
    for t in tickers:
        try:
            data = yf.Ticker(t).history(period=period, auto_adjust=True)
            if data is not None and not data.empty:
                prices_dict[t] = data["Close"]
        except Exception:
            continue

    if not prices_dict:
        return None

    prices_df = pd.concat(prices_dict, axis=1)
    prices_df.columns = list(prices_df.columns)
    prices_df = prices_df.dropna(how="all")
    return prices_df


def get_benchmark_price_data(
    benchmark_ticker: str,
    period: str = "3y",
) -> Optional[pd.DataFrame]:
    """
    Descarga precios ajustados diarios para un benchmark (ticker único).

    Parameters
    ----------
    benchmark_ticker : str
        Ticker del benchmark.
    period : str
        Periodo, por ejemplo '1y', '3y', '10y'.

    Returns
    -------
    Optional[pd.DataFrame]
        DataFrame con una sola columna de precios de cierre ajustados.
    """
    if not benchmark_ticker:
        return None

    try:
        data = yf.Ticker(benchmark_ticker).history(period=period, auto_adjust=True)
        if data is None or data.empty:
            return None
        df = data[["Close"]].rename(columns={"Close": benchmark_ticker})
        df = df.dropna()
        return df
    except Exception:
        return None
