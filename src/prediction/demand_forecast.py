"""Prediccion de demanda por celda H3 como insumo para la optimizacion.

Complementa al motor VRP (que optimiza rutas dada la demanda observada) con
un modelo que estima la demanda esperada de una celda H3 a partir solo de
su ubicacion (comuna + coordenadas del centroide) -- util para evaluar
candidatas a Dark Store en celdas donde todavia no hay pedidos historicos.

Se comparan tres enfoques, todos alimentando la misma optimizacion aguas
abajo (no la reemplazan):

1. Baseline interpretable: regresion Ridge sobre features espaciales.
2. Ensamble de arboles: RandomForestRegressor.
3. Red neuronal (PyTorch): MLP con una loss custom asimetrica -- penaliza
   mas la subestimacion de demanda que la sobreestimacion (un stockout en
   un Dark Store cuesta mas que sobre-aprovisionar) -- comparando las
   activaciones ReLU, GELU y Swish (SiLU).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import geopandas as gpd
except ImportError:  # pragma: no cover
    gpd = None

import torch
from torch import nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = Path(__file__).resolve().parents[2] / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
METRICS_DB = OUTPUTS_DIR / "metrics" / "demand_forecast_metrics.duckdb"

SEED = 42
ACTIVATIONS = ("relu", "gelu", "swish")


# ---------------------------------------------------------------------------
# Datos y features
# ---------------------------------------------------------------------------

def load_h3_dataset(path: Path | None = None) -> pd.DataFrame:
    """Carga la agregacion de demanda por celda H3 (`aggregate_by_h3`)."""
    path = path or (PROCESSED_DIR / "h3_demand_agg.geojson")
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontro {path}. Corre primero `python -m src.spatial.geo_generator`."
        )
    if gpd is not None:
        gdf = gpd.read_file(path)
        return pd.DataFrame(gdf.drop(columns="geometry"))
    return pd.read_json(path)  # pragma: no cover - fallback sin geopandas


def build_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Construye la matriz de features (solo ubicacion) y el target (`total_demand`).

    Deliberadamente NO se usa `order_count` como feature: en un candidato a
    Dark Store sin historial, tampoco se conoceria de antemano. Las features
    son las que si estarian disponibles para cualquier celda H3 candidata:
    su comuna y las coordenadas de su centroide.
    """
    comuna_dummies = pd.get_dummies(df["comuna"], prefix="comuna")
    X_df = pd.concat([df[["centroid_lat", "centroid_lon"]], comuna_dummies], axis=1)
    feature_names = list(X_df.columns)
    X = X_df.to_numpy(dtype=np.float64)
    y = df["total_demand"].to_numpy(dtype=np.float64)
    return X, y, feature_names


@dataclass
class Split:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


def make_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.25, seed: int = SEED) -> Split:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    return Split(X_train, X_test, y_train, y_test)


# ---------------------------------------------------------------------------
# Metricas
# ---------------------------------------------------------------------------

def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    return {"mae": float(mae), "rmse": rmse, "r2": float(r2)}


# ---------------------------------------------------------------------------
# 1. Baseline interpretable -- Ridge
# ---------------------------------------------------------------------------

def train_baseline(split: Split, alpha: float = 1.0) -> Ridge:
    model = Ridge(alpha=alpha, random_state=SEED)
    model.fit(split.X_train, split.y_train)
    return model


# ---------------------------------------------------------------------------
# 2. Ensamble -- Random Forest
# ---------------------------------------------------------------------------

def train_random_forest(split: Split, n_estimators: int = 200) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=n_estimators, random_state=SEED, max_depth=6, min_samples_leaf=2
    )
    model.fit(split.X_train, split.y_train)
    return model


# ---------------------------------------------------------------------------
# 3. Red neuronal (PyTorch) -- MLP con loss asimetrica, comparando activaciones
# ---------------------------------------------------------------------------

def _activation_layer(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "swish":
        return nn.SiLU()  # Swish == SiLU (x * sigmoid(x))
    raise ValueError(f"Activacion desconocida: {name}")


class DemandMLP(nn.Module):
    """MLP pequeno para regresion de demanda por celda H3."""

    def __init__(self, n_features: int, activation: str = "relu", hidden: int = 32):
        super().__init__()
        act = _activation_layer(activation)
        act2 = _activation_layer(activation)
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            act,
            nn.Linear(hidden, hidden // 2),
            act2,
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class AsymmetricStockoutLoss(nn.Module):
    """Loss custom: penaliza mas la subestimacion (`pred < y`) que la
    sobreestimacion, porque en el dominio (dimensionamiento de un Dark
    Store / flota) un stockout de demanda no cubierta es mas costoso que
    sobre-aprovisionar capacidad ociosa.
    """

    def __init__(self, underprediction_weight: float = 2.5):
        super().__init__()
        self.w = underprediction_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        error = pred - target
        weight = torch.where(error < 0, self.w, 1.0)
        return torch.mean(weight * error**2)


def train_mlp(
    split: Split,
    activation: str,
    epochs: int = 300,
    lr: float = 0.01,
    underprediction_weight: float = 2.5,
    seed: int = SEED,
) -> tuple[DemandMLP, StandardScaler, list[float]]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(split.X_train)

    X_t = torch.tensor(X_train_s, dtype=torch.float32)
    y_t = torch.tensor(split.y_train, dtype=torch.float32)

    model = DemandMLP(n_features=split.X_train.shape[1], activation=activation)
    criterion = AsymmetricStockoutLoss(underprediction_weight=underprediction_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history: list[float] = []
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(X_t)
        loss = criterion(pred, y_t)
        loss.backward()
        optimizer.step()
        history.append(float(loss.item()))

    return model, scaler, history


def predict_mlp(model: DemandMLP, scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    model.eval()
    X_s = scaler.transform(X)
    with torch.no_grad():
        pred = model(torch.tensor(X_s, dtype=torch.float32))
    return pred.numpy()


# ---------------------------------------------------------------------------
# Orquestacion
# ---------------------------------------------------------------------------

def run_comparison(df: pd.DataFrame | None = None) -> dict:
    """Entrena los 3 enfoques (+ 3 activaciones del MLP) y devuelve metricas,
    predicciones e historiales de loss, listos para graficar/persistir."""
    df = df if df is not None else load_h3_dataset()
    X, y, feature_names = build_features(df)
    split = make_split(X, y)

    results: dict = {"feature_names": feature_names, "n_samples": len(df), "models": {}}

    ridge = train_baseline(split)
    results["models"]["ridge_baseline"] = {
        "metrics": evaluate(split.y_test, ridge.predict(split.X_test)),
        "y_pred": ridge.predict(split.X_test),
    }

    rf = train_random_forest(split)
    results["models"]["random_forest"] = {
        "metrics": evaluate(split.y_test, rf.predict(split.X_test)),
        "y_pred": rf.predict(split.X_test),
    }

    loss_histories: dict[str, list[float]] = {}
    for activation in ACTIVATIONS:
        model, scaler, history = train_mlp(split, activation=activation)
        y_pred = predict_mlp(model, scaler, split.X_test)
        key = f"mlp_{activation}"
        results["models"][key] = {
            "metrics": evaluate(split.y_test, y_pred),
            "y_pred": y_pred,
        }
        loss_histories[activation] = history

    results["loss_histories"] = loss_histories
    results["y_test"] = split.y_test
    return results


# ---------------------------------------------------------------------------
# Persistencia en DuckDB
# ---------------------------------------------------------------------------

def persist_metrics_duckdb(results: dict, db_path: Path | None = None, run_label: str = "default") -> Path:
    import duckdb

    db_path = db_path or METRICS_DB
    db_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(db_path))
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS demand_forecast_metrics (
            run_label VARCHAR,
            model_name VARCHAR,
            mae DOUBLE,
            rmse DOUBLE,
            r2 DOUBLE,
            n_samples INTEGER,
            created_at TIMESTAMP DEFAULT current_timestamp
        )
        """
    )
    con.execute("DELETE FROM demand_forecast_metrics WHERE run_label = ?", [run_label])
    for model_name, payload in results["models"].items():
        m = payload["metrics"]
        con.execute(
            "INSERT INTO demand_forecast_metrics (run_label, model_name, mae, rmse, r2, n_samples) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [run_label, model_name, m["mae"], m["rmse"], m["r2"], results["n_samples"]],
        )
    con.close()
    return db_path


def read_metrics_duckdb(db_path: Path | None = None) -> pd.DataFrame:
    import duckdb

    db_path = db_path or METRICS_DB
    con = duckdb.connect(str(db_path))
    df = con.execute("SELECT * FROM demand_forecast_metrics ORDER BY run_label, model_name").fetchdf()
    con.close()
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    from src.prediction.plots import plot_all

    df = load_h3_dataset()
    results = run_comparison(df)

    for name, payload in results["models"].items():
        m = payload["metrics"]
        print(f"{name:16s} MAE={m['mae']:.3f}  RMSE={m['rmse']:.3f}  R2={m['r2']:.3f}")

    db_path = persist_metrics_duckdb(results)
    print(f"\nMetricas persistidas en: {db_path}")

    plot_paths = plot_all(results)
    for p in plot_paths:
        print(f"Grafico guardado en: {p}")


if __name__ == "__main__":
    main()
