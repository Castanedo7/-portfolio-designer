import os
from datetime import datetime
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

import google.generativeai as genai



from modules.data_loader import get_price_data, get_benchmark_price_data
from modules.portfolio_math import (
    compute_returns,
    get_portfolio_timeseries,
    compute_basic_stats,
    compute_beta,
)
from modules.efficient_frontier import simulate_efficient_frontier
from modules.comparisons import build_comparison_df
from modules.ai_module import build_ai_payload, run_gemini_analysis
from modules.utils import normalize_weights_safe, format_percentage

# -------------------------------
# Configuración general Streamlit
# -------------------------------

st.set_page_config(
    page_title="Portfolio Designer",
    page_icon="📊",
    layout="wide",
)

# -------------------------------
# Cargar variables de entorno
# -------------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    GEMINI_MODEL_NAME = "gemini-2.5-flash"
else:
    GEMINI_MODEL_NAME = None

# -------------------------------
# Sidebar: Config global
# -------------------------------

st.sidebar.title("⚙️ Configuración global")

risk_free_rate = st.sidebar.number_input(
    "Tasa libre de riesgo anual (ej. 0.04 = 4%)",
    min_value=-0.05,
    max_value=0.20,
    value=0.04,
    step=0.01,
    format="%.4f",
)

benchmark_ticker = st.sidebar.text_input(
    "Benchmark para beta",
    value="SPY",
    help="Ticker del índice de referencia para calcular beta del portafolio.",
)

st.sidebar.markdown("---")
st.sidebar.caption("Portfolio Designer · Python + Streamlit + Gemini")


# -------------------------------
# Función principal
# -------------------------------

def main() -> None:
    st.title("📊 Portfolio Designer")
    st.markdown(
        "Diseña, analiza y compara portafolios de hasta **10 activos** con frontera eficiente y análisis con IA (Gemini)."
    )

    (
        config_tab,
        performance_tab,
        frontier_tab,
        compare_tab,
        ai_tab,
        about_tab,
    ) = st.tabs(
        [
            "🎚 Configuración",
            "📈 Desempeño histórico",
            "🧮 Frontera eficiente",
            "⚖️ Comparador vs Benchmarks",
            "🤖 Análisis con IA",
            "ℹ️ Acerca de",
        ]
    )

    # ---------------------------
    # 1) CONFIGURACIÓN
    # ---------------------------
    with config_tab:
        (
            tickers,
            weights,
            period_str,
            data_ok,
            prices_df,
            returns_df,
            portfolio_returns,
            bench_returns_for_beta,
        ) = render_configuration_tab()

    # Si la config no es válida, detenemos el resto
    if not data_ok:
        with performance_tab:
            st.info("Configura un portafolio válido en la pestaña **Configuración**.")
        with frontier_tab:
            st.info("Configura un portafolio válido en la pestaña **Configuración**.")
        with compare_tab:
            st.info("Configura un portafolio válido en la pestaña **Configuración**.")
        with ai_tab:
            st.info("Configura un portafolio válido en la pestaña **Configuración**.")
        with about_tab:
            render_about_tab()
        return

    # ---------------------------
    # Cálculos base compartidos
    # ---------------------------
    mean_returns = returns_df.mean()
    cov_matrix = returns_df.cov()

    basic_stats = compute_basic_stats(
        portfolio_returns=portfolio_returns,
        risk_free_rate=risk_free_rate,
    )

    beta_value = compute_beta(
        portfolio_returns=portfolio_returns,
        benchmark_returns=bench_returns_for_beta,
    )

    # ---------------------------
    # 2) DESEMPEÑO HISTÓRICO
    # ---------------------------
    with performance_tab:
        render_performance_tab(
            tickers=tickers,
            weights=weights,
            prices_df=prices_df,
            portfolio_returns=portfolio_returns,
            basic_stats=basic_stats,
            beta_value=beta_value,
        )

    # ---------------------------
    # 3) FRONTERA EFICIENTE
    # ---------------------------
    with frontier_tab:
        frontier_results = render_frontier_tab(
            mean_returns=mean_returns,
            cov_matrix=cov_matrix,
            risk_free_rate=risk_free_rate,
            tickers=tickers,
        )

    # ---------------------------
    # 4) COMPARADOR
    # ---------------------------
    with compare_tab:
        comparison_df = render_comparison_tab(
            portfolio_returns=portfolio_returns,
            base_prices_df=prices_df,
        )

    # ---------------------------
    # 5) IA (Gemini)
    # ---------------------------
    with ai_tab:
        render_ai_tab(
            tickers=tickers,
            weights=weights,
            basic_stats=basic_stats,
            beta_value=beta_value,
            frontier_results=frontier_results,
            comparison_df=comparison_df,
        )

    # ---------------------------
    # 6) ACERCA DE
    # ---------------------------
    with about_tab:
        render_about_tab()


# -------------------------------
# Render: Configuración
# -------------------------------

def render_configuration_tab():
    st.subheader("🎚 Configuración del portafolio")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("#### Activos y pesos")

        n_assets = st.number_input(
            "Número de activos",
            min_value=1,
            max_value=10,
            value=3,
            step=1,
        )

        tickers: List[str] = []
        raw_weights: List[float] = []

        for i in range(int(n_assets)):
            c1, c2 = st.columns([2, 1])
            with c1:
                ticker = st.text_input(
                    f"Ticker {i + 1}",
                    key=f"ticker_{i}",
                    help="Ejemplo: AAPL, MSFT, IVV, NAFTRAC.MX",
                ).strip().upper()
            with c2:
                w = st.number_input(
                    f"Peso {i + 1}",
                    min_value=0.0,
                    max_value=1.0,
                    value=round(1.0 / n_assets, 2),
                    step=0.01,
                    key=f"weight_{i}",
                )

            if ticker:
                tickers.append(ticker)
                raw_weights.append(w)

        if len(tickers) == 0:
            st.warning("Ingresa al menos **un ticker** para el portafolio.")
            return [], [], "1y", False, None, None, None, None

        weights = normalize_weights_safe(raw_weights)

        st.info(
            "Los pesos han sido **normalizados automáticamente** para sumar 1.0.\n\n"
            + "\n".join(
                [
                    f"- {t}: {format_percentage(w)}"
                    for t, w in zip(tickers, weights)
                ]
            )
        )

    with col_right:
        st.markdown("#### Horizonte histórico")

        period_choice = st.radio(
            "Periodo de análisis",
            options=["1 año", "3 años", "10 años"],
            index=0,
            horizontal=True,
        )

        period_map = {
            "1 año": "1y",
            "3 años": "3y",
            "10 años": "10y",
        }
        period_str = period_map[period_choice]

        st.caption(
            "Se usan datos diarios ajustados. Si un activo no tiene historial suficiente, "
            "podría excluirse automáticamente."
        )

    # Descargar precios
    with st.spinner("Descargando precios históricos..."):
        prices_df = get_price_data(tickers=tickers, period=period_str)

    if prices_df is None or prices_df.empty:
        st.error("No se pudieron obtener datos de precios para los tickers ingresados.")
        return tickers, weights, period_str, False, None, None, None, None

    returns_df = compute_returns(prices_df)

    # Benchmark para beta (mismo periodo)
    benchmark_prices = get_benchmark_price_data(
        benchmark_ticker=benchmark_ticker, period=period_str
    )
    if benchmark_prices is not None and not benchmark_prices.empty:
        bench_returns = compute_returns(benchmark_prices)
    else:
        bench_returns = None

    portfolio_returns = get_portfolio_timeseries(
        returns_df=returns_df,
        weights=np.array(weights),
    )

    st.success("✅ Datos descargados y portafolio configurado correctamente.")

    st.markdown("##### Vista rápida de precios (últimas filas)")
    st.dataframe(prices_df.tail())

    return (
        tickers,
        weights,
        period_str,
        True,
        prices_df,
        returns_df,
        portfolio_returns,
        bench_returns,
    )


# -------------------------------
# Render: Desempeño histórico
# -------------------------------

def render_performance_tab(
    tickers: List[str],
    weights: List[float],
    prices_df: pd.DataFrame,
    portfolio_returns: pd.Series,
    basic_stats: Dict[str, float],
    beta_value: Optional[float],
) -> None:
    st.subheader("📈 Desempeño histórico del portafolio")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rend. anualizado", format_percentage(basic_stats["annual_return"]))
    col2.metric("Volatilidad anualizada", format_percentage(basic_stats["annual_volatility"]))
    col3.metric("Sharpe Ratio", f"{basic_stats['sharpe_ratio']:.2f}")
    if beta_value is not None:
        col4.metric("Beta vs benchmark", f"{beta_value:.2f}")
    else:
        col4.metric("Beta vs benchmark", "N/D")

    # Serie de portafolio
    cum_portfolio = (1 + portfolio_returns).cumprod()

    # Gráfica portafolio
    st.markdown("#### Rendimiento acumulado del portafolio")
    fig_port = px.line(
        cum_portfolio,
        x=cum_portfolio.index,
        y=cum_portfolio.values,
        labels={"x": "Fecha", "y": "Rendimiento acumulado"},
        title="Portafolio (100 = valor inicial)",
    )
    fig_port.update_traces(name="Portafolio", showlegend=True)
    st.plotly_chart(fig_port, use_container_width=True)

    # Gráfica de activos individuales (normalizados)
    st.markdown("#### Activos del portafolio (normalizados)")
    norm_prices = prices_df / prices_df.iloc[0] * 100.0
    fig_assets = px.line(
        norm_prices,
        x=norm_prices.index,
        y=norm_prices.columns,
        labels={"value": "Índice (100 = inicio)", "variable": "Ticker", "x": "Fecha"},
        title="Activos normalizados",
    )
    st.plotly_chart(fig_assets, use_container_width=True)

    st.markdown("#### Detalle de activos y pesos")
    detail_df = pd.DataFrame(
        {
            "Ticker": tickers,
            "Peso": weights,
        }
    )
    detail_df["Peso (%)"] = detail_df["Peso"].apply(lambda x: round(x * 100, 2))
    st.dataframe(detail_df, hide_index=True)


# -------------------------------
# Render: Frontera eficiente
# -------------------------------

def render_frontier_tab(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float,
    tickers: List[str],
) -> Dict[str, Any]:
    st.subheader("🧮 Frontera eficiente")

    num_portfolios = st.slider(
        "Número de portafolios simulados",
        min_value=1000,
        max_value=20000,
        value=5000,
        step=1000,
        help="Más portafolios = mejor aproximación pero más tiempo de cómputo.",
    )

    with st.spinner("Simulando portafolios aleatorios..."):
        frontier_results = simulate_efficient_frontier(
            mean_returns=mean_returns,
            cov_matrix=cov_matrix,
            risk_free_rate=risk_free_rate,
            num_portfolios=num_portfolios,
        )

    df_frontier = frontier_results["frontier_df"]

    st.markdown("#### Frontera eficiente (riesgo vs rendimiento)")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df_frontier["risk"],
            y=df_frontier["return"],
            mode="markers",
            name="Portafolios simulados",
            opacity=0.5,
        )
    )

    # Portafolios clave
    for key, name, color in [
        ("min_risk_portfolio", "Mínimo riesgo", "blue"),
        ("max_return_portfolio", "Máximo rendimiento", "orange"),
        ("max_sharpe_portfolio", "Máximo Sharpe", "green"),
    ]:
        p = frontier_results[key]
        fig.add_trace(
            go.Scatter(
                x=[p["risk"]],
                y=[p["return"]],
                mode="markers",
                marker=dict(size=12),
                name=name,
            )
        )

    fig.update_layout(
        xaxis_title="Riesgo (volatilidad anualizada)",
        yaxis_title="Rendimiento anualizado",
        legend_title="Portafolios",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Detalle de portafolios óptimos")

    cols = st.columns(3)
    labels = ["Mínimo riesgo", "Máximo rendimiento", "Máximo Sharpe"]
    keys = ["min_risk_portfolio", "max_return_portfolio", "max_sharpe_portfolio"]

    for col, label, key in zip(cols, labels, keys):
        p = frontier_results[key]
        with col:
            st.markdown(f"**{label}**")
            st.write(f"Rend. anualizado: {format_percentage(p['return'])}")
            st.write(f"Riesgo (vol): {format_percentage(p['risk'])}")
            st.write(f"Sharpe: {p['sharpe']:.2f}")

    st.markdown("#### Pesos del portafolio de máximo Sharpe")
    max_sharpe_weights = frontier_results["max_sharpe_portfolio"]["weights"]
    weights_df = pd.DataFrame(
        {
            "Ticker": tickers,
            "Peso óptimo": max_sharpe_weights,
            "Peso óptimo (%)": [round(w * 100, 2) for w in max_sharpe_weights],
        }
    )
    st.dataframe(weights_df, hide_index=True)

    return frontier_results


# -------------------------------
# Render: Comparador vs Benchmarks
# -------------------------------

def render_comparison_tab(
    portfolio_returns: pd.Series,
    base_prices_df: pd.DataFrame,
) -> pd.DataFrame:
    st.subheader("⚖️ Comparador contra benchmarks")

    st.markdown(
        "Se compara el portafolio (por ejemplo, el de máximo Sharpe o el actual) contra benchmarks estándar."
    )

    default_benchmarks = ["QQQ", "SPY", "GLD"]
    extra_ticker = st.text_input(
        "Ticker adicional para comparar (opcional)",
        value="",
        help="Por ejemplo: ^GSPC, NAFTRAC.MX, etc.",
    ).strip().upper()

    benchmarks = default_benchmarks.copy()
    if extra_ticker:
        benchmarks.append(extra_ticker)

    with st.spinner("Descargando benchmarks y construyendo comparación..."):
        comparison_df = build_comparison_df(
            portfolio_returns=portfolio_returns,
            benchmark_tickers=benchmarks,
        )

    if comparison_df is None or comparison_df.empty:
        st.error("No fue posible construir la comparación con benchmarks.")
        return pd.DataFrame()

    st.markdown("#### Rendimiento acumulado comparativo (100 = inicio)")
    fig = px.line(
        comparison_df,
        x=comparison_df.index,
        y=comparison_df.columns,
        labels={"value": "Índice (100 = inicio)", "variable": "Serie", "x": "Fecha"},
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Rendimiento final (ordenado)")
    final_values = comparison_df.iloc[-1].sort_values(ascending=False)
    final_df = final_values.to_frame(name="Índice final").reset_index()
    final_df.rename(columns={"index": "Serie"}, inplace=True)
    st.dataframe(final_df, hide_index=True)

    return comparison_df


# -------------------------------
# Render: IA con Gemini
# -------------------------------

def render_ai_tab(
    tickers: List[str],
    weights: List[float],
    basic_stats: Dict[str, float],
    beta_value: Optional[float],
    frontier_results: Dict[str, Any],
    comparison_df: pd.DataFrame,
) -> None:
    st.subheader("🤖 Análisis con IA (Gemini)")

    if GEMINI_MODEL_NAME is None:
        st.error(
            "No se encontró `GEMINI_API_KEY` en el entorno. "
            "Configura tu archivo `.env` para habilitar el análisis con IA."
        )
        return

    st.markdown(
        "La IA actuará como un **gestor de portafolio profesional**, "
        "analizando concentración, riesgo, diversificación y comparativos."
    )

    if st.button("Generar análisis con IA"):
        with st.spinner("Llamando a Gemini y generando análisis..."):
            payload = build_ai_payload(
                tickers=tickers,
                weights=weights,
                basic_stats=basic_stats,
                beta_value=beta_value,
                frontier_results=frontier_results,
                comparison_df=comparison_df,
            )
            analysis_text = run_gemini_analysis(
                model_name=GEMINI_MODEL_NAME,
                payload=payload,
            )

        if analysis_text:
            st.markdown("#### Resultado del análisis")
            st.write(analysis_text)
        else:
            st.error("Ocurrió un problema al obtener el análisis de Gemini.")
    else:
        st.info("Pulsa el botón para generar el análisis.")


# -------------------------------
# Render: Acerca de
# -------------------------------

def render_about_tab() -> None:
    st.subheader("ℹ️ Acerca de Portfolio Designer")
    st.markdown(
        """
**Portfolio Designer** es una app construida en Python + Streamlit para:

- Diseñar portafolios de hasta 10 activos.
- Ver su desempeño histórico (rend., volatilidad, Sharpe, beta).
- Simular la frontera eficiente con miles de portafolios.
- Comparar contra benchmarks como QQQ, SPY y GLD.
- Recibir un análisis en lenguaje natural usando Gemini (modelo `gemini-2.5-flash`).

Código estructurado para subir a GitHub, con:
- Entorno virtual recomendado: `.venv`
- Lectura de `GEMINI_API_KEY` vía `python-dotenv`.
"""
    )


if __name__ == "__main__":
    main()
