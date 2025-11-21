from typing import List


def normalize_weights_safe(weights: List[float]) -> List[float]:
    """
    Normaliza una lista de pesos para que sumen 1.0.
    Si la suma es 0, reparte equitativamente.

    Parameters
    ----------
    weights : List[float]
        Pesos originales (pueden no sumar 1).

    Returns
    -------
    List[float]
        Pesos normalizados.
    """
    total = sum(weights)
    n = len(weights)
    if n == 0:
        return []

    if total <= 0:
        # Si todo es cero o negativo, repartir equitativamente
        return [1.0 / n] * n

    return [w / total for w in weights]


def format_percentage(value: float) -> str:
    """
    Da formato de porcentaje con dos decimales.
    """
    return f"{value * 100:.2f}%"
