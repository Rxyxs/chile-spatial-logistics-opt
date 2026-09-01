"""Graficos explicativos para la comparacion de modelos de prediccion de demanda.

Sigue la paleta y estilo matplotlib ya usados por el resto del proyecto
(colores solidos, `bbox_inches="tight"`, salida a `outputs/plots/`).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

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


def plot_activation_curves_animated(results: dict, out_path: Path | None = None) -> Path:
    """Version animada (GIF) de la convergencia del MLP: linea de carrera con
    etiqueta flotante que muestra la loss actual de cada activacion.

    Usa exactamente los mismos `loss_histories` reales que
    `plot_activation_curves`; solo se subdivide la cantidad de cuadros.
    """
    out_path = out_path or (PLOTS_DIR / "mlp_activation_convergence_animated.gif")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    histories = {act: list(h) for act, h in results["loss_histories"].items()}
    n_epochs = max(len(h) for h in histories.values())

    n_frames = min(50, n_epochs)
    # Indices reales (sin inventar valores) subsampleados a lo sumo a n_frames.
    frame_epochs = sorted(set(np.linspace(1, n_epochs, n_frames, dtype=int)))

    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_xlim(1, n_epochs)
        all_losses = [v for h in histories.values() for v in h]
        ax.set_ylim(min(all_losses) * 0.95, max(all_losses) * 1.05)
        ax.set_xlabel("Epoca")
        ax.set_ylabel("Loss (asimetrica de stockout)")
        ax.set_title("Convergencia del MLP por funcion de activacion (animado)")

        lines = {}
        labels = {}
        for activation, history in histories.items():
            color = ACTIVATION_COLORS.get(activation, "#33cccc")
            (line,) = ax.plot([], [], color=color, linewidth=2, label=activation)
            lines[activation] = line
            labels[activation] = ax.annotate(
                "",
                xy=(1, history[0]),
                xytext=(10, 0),
                textcoords="offset points",
                color="black",
                fontsize=9,
                fontweight="bold",
                va="center",
                bbox=dict(boxstyle="round,pad=0.3", fc=color, ec="none", alpha=0.9),
            )
        ax.legend(loc="upper right")
        fig.tight_layout()

        def update(frame_idx):
            epoch = frame_epochs[frame_idx]
            artists = []
            for activation, history in histories.items():
                x = list(range(1, epoch + 1))
                y = history[:epoch]
                lines[activation].set_data(x, y)
                current = history[epoch - 1]
                labels[activation].set_position((10, 0))
                labels[activation].xy = (epoch, current)
                labels[activation].set_text(f"{activation}: {current:.3f}")
                artists.append(lines[activation])
                artists.append(labels[activation])
            return artists

        ani = FuncAnimation(fig, update, frames=len(frame_epochs), interval=120, blit=False)
        ani.save(out_path, writer="pillow")
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
        plot_activation_curves_animated(results),
        plot_actual_vs_predicted(results),
    ]
