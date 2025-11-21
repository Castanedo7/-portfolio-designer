from typing import Dict, Any, Optional

import json
import os
import numpy as np
import pandas as pd
import google.generativeai as genai


# ---------------------------------------------------------
#   Función para convertir todo el payload a JSON válido
# ---------------------------------------------------------
def make_json_serializable(obj):
    """
    Convierte numpy arrays, numpy floats, DataFrames
    y otros objetos a tipos compatibles con JSON.
    """
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}

    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]

    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")

    elif isinstance(obj, pd.Series):
        return obj.to_dict()

    elif hasattr(obj, "tolist"):  # numpy arrays
        return obj.tolist()

    elif isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
        return obj.item()

    else:
        return obj


# ---------------------------------------------------------
#   Construcción de payload para la IA
# ---------------------------------------------------------
def build_ai_payload(
    tickers: list,
    weights: list,
    basic_stats: Dict[str, float],
    beta_value: Optional[float],
    frontier_results: Dict[str, Any],
    comparison_df: pd.DataFrame,
) -> Dict[str, Any]:

    frontier_df = frontier_results["frontier_df"]

    payload = {
        "portfolio": {
            "tickers": tickers,
            "weights": list(weights),
            "weights_optimal_max_sharpe": list(frontier_results["max_sharpe_portfolio"]["weights"]),
            "basic_stats": basic_stats,
            "beta_vs_benchmark": (
                float(beta_value) if beta_value is not None else None
            ),
        },
        "efficient_frontier": {
            "min_risk_portfolio": make_json_serializable(frontier_results["min_risk_portfolio"]),
            "max_return_portfolio": make_json_serializable(frontier_results["max_return_portfolio"]),
            "max_sharpe_portfolio": make_json_serializable(frontier_results["max_sharpe_portfolio"]),
            "frontier_sample_rows": frontier_df.head(20).to_dict(orient="records"),
        },
    }

    if comparison_df is not None and not comparison_df.empty:
        payload["comparisons"] = {
            "end_date": str(comparison_df.index[-1].date()),
            "final_index_values": comparison_df.iloc[-1].to_dict(),
        }
    else:
        payload["comparisons"] = None

    return payload


# ---------------------------------------------------------
#   Función de análisis con Gemini
# ---------------------------------------------------------
def run_gemini_analysis(model_name: str, payload: Dict[str, Any]) -> Optional[str]:
    """
    Llama a Gemini con un prompt diseñado para análisis de portafolio profesional.
    """

    # Configurar API key
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

    # Crear modelo
    try:
        model = genai.GenerativeModel(model_name)
    except Exception:
        return "Error: No se pudo inicializar el modelo de Gemini."

    # Convertir payload a JSON
    try:
        payload = make_json_serializable(payload)
        payload_text = json.dumps(payload, indent=2)
    except Exception as e:
        return f"Error al serializar JSON: {e}"

    # Crear prompts
    system_prompt = (
        "Actúa como un gestor de portafolio profesional.\n"
        "Analiza concentración, riesgo, volatilidad, Sharpe, beta, diversificación,\n"
        "comparaciones con benchmarks, y mejoras posibles al portafolio.\n"
        "Responde en español, con secciones claras y enfoque cuantitativo.\n"
    )

    user_prompt = (
        "Aquí tienes los datos del portafolio en JSON:\n\n"
        f"{payload_text}\n\n"
        "Por favor realiza un análisis completo según las instrucciones."
    )

    # Nuevo formato correcto para Gemini (solo rol 'user')
    full_prompt = system_prompt + "\n\n" + user_prompt

    try:
        response = model.generate_content(
            [
                {"role": "user", "parts": [full_prompt]}
            ]
        )

        if hasattr(response, "text"):
            return response.text

        return "Error: La respuesta no contiene texto."
    except Exception as e:
        return f"Error al llamar a Gemini: {e}"
