"""Graficos explicativos para la comparacion de modelos de prediccion de demanda.

Sigue la paleta y estilo matplotlib ya usados por el resto del proyecto
(colores solidos, `bbox_inches="tight"`, salida a `outputs/plots/`).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PLOTS_DIR = Path(__file__).resolve().parents[2] / "outputs" / "plots"

# Paleta consistente por modelo/activacion.
MODEL_COLORS = {
    "ridge_baseline": "#4C72B0",
    "random_forest": "#55A868",
    "mlp_relu": "#C44E52",
    "mlp_gelu": "#8172B2",
    "mlp_swish": "#CCB974",
}

ACTIVATION_COLORS = {
    "relu": "#C44E52",
    "gelu": "#8172B2",
    "swish": "#CCB974",
}


def plot_model_comparison(results: dict, out_path: Path | None = None) -> Path:
    """Barras comparando MAE y RMSE entre los 3 enfoques (+3 activaciones del MLP)."""
    out_path = out_path or (PLOTS_DIR / "demand_model_comparison.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    names = list(results["models"].keys())
    mae = [results["models"][n]["metrics"]["mae"] for n in names]
    rmse = [results["models"][n]["metrics"]["rmse"] for n in names]
    colors = [MODEL_COLORS.get(n, "#999999") for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(len(names))

    axes[0].bar(x, mae, color=colors)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=30, ha="right")
    axes[0].set_ylabel("MAE (unidades de demanda)")
    axes[0].set_title("Error absoluto medio por modelo")

    axes[1].bar(x, rmse, color=colors)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=30, ha="right")
    axes[1].set_ylabel("RMSE (unidades de demanda)")
    axes[1].set_title("Raiz del error cuadratico medio por modelo")

    fig.suptitle("Prediccion de demanda por celda H3 -- comparacion de enfoques")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_activation_curves(results: dict, out_path: Path | None = None) -> Path:
    """Curvas de convergencia (loss por epoca) del MLP, una linea por activacion."""
    out_path = out_path or (PLOTS_DIR / "mlp_activation_convergence.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for activation, history in results["loss_histories"].items():
        ax.plot(history, label=activation, color=ACTIVATION_COLORS.get(activation, "#333333"))

    ax.set_xlabel("Epoca")
    ax.set_ylabel("Loss (asimetrica de stockout)")
    ax.set_title("Convergencia del MLP por funcion de activacion")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_actual_vs_predicted(results: dict, out_path: Path | None = None) -> Path:
    """Dispersión real vs. predicho para el mejor modelo (menor MAE)."""
    out_path = out_path or (PLOTS_DIR / "demand_actual_vs_predicted.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best_name = min(results["models"], key=lambda n: results["models"][n]["metrics"]["mae"])
    y_test = results["y_test"]
    y_pred = results["models"][best_name]["y_pred"]

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y_test, y_pred, alpha=0.6, color=MODEL_COLORS.get(best_name, "#4C72B0"))
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, linestyle="--", color="#888888", label="prediccion perfecta")
    ax.set_xlabel("Demanda real (total_demand)")
    ax.set_ylabel("Demanda predicha")
    ax.set_title(f"Real vs. predicho -- mejor modelo: {best_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_all(results: dict) -> list[Path]:
    return [
        plot_model_comparison(results),
        plot_activation_curves(results),
        plot_actual_vs_predicted(results),
    ]
