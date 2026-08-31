from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.prediction.demand_forecast import (
    AsymmetricStockoutLoss,
    DemandMLP,
    build_features,
    evaluate,
    make_split,
    persist_metrics_duckdb,
    predict_mlp,
    read_metrics_duckdb,
    run_comparison,
    train_baseline,
    train_mlp,
    train_random_forest,
)


@pytest.fixture(scope="module")
def synthetic_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 60
    comunas = rng.choice(["Las Condes", "Providencia", "Maipu"], size=n)
    lat = rng.uniform(-33.6, -33.4, size=n)
    lon = rng.uniform(-70.7, -70.5, size=n)
    demand = rng.integers(1, 30, size=n).astype(float)
    return pd.DataFrame(
        {
            "h3_cell": [f"cell_{i}" for i in range(n)],
            "comuna": comunas,
            "centroid_lat": lat,
            "centroid_lon": lon,
            "total_demand": demand,
            "order_count": rng.integers(1, 10, size=n),
        }
    )


def test_build_features_shapes(synthetic_df):
    X, y, names = build_features(synthetic_df)
    assert X.shape[0] == len(synthetic_df)
    assert y.shape[0] == len(synthetic_df)
    # 2 coords + one dummy column per comuna
    assert X.shape[1] == 2 + synthetic_df["comuna"].nunique()
    assert "order_count" not in names  # no debe fugarse info no disponible en un candidato nuevo


def test_evaluate_perfect_prediction_has_zero_error():
    y = np.array([1.0, 2.0, 3.0])
    metrics = evaluate(y, y)
    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["rmse"] == pytest.approx(0.0)
    assert metrics["r2"] == pytest.approx(1.0)


def test_ridge_baseline_trains_and_predicts(synthetic_df):
    X, y, _ = build_features(synthetic_df)
    split = make_split(X, y)
    model = train_baseline(split)
    preds = model.predict(split.X_test)
    assert preds.shape == split.y_test.shape
    assert np.all(np.isfinite(preds))


def test_random_forest_trains_and_predicts(synthetic_df):
    X, y, _ = build_features(synthetic_df)
    split = make_split(X, y)
    model = train_random_forest(split, n_estimators=20)
    preds = model.predict(split.X_test)
    assert preds.shape == split.y_test.shape


def test_asymmetric_loss_penalizes_underprediction_more():
    loss = AsymmetricStockoutLoss(underprediction_weight=3.0)
    target = torch.tensor([10.0])
    under = torch.tensor([5.0])   # subestima en 5
    over = torch.tensor([15.0])   # sobreestima en 5
    loss_under = loss(under, target)
    loss_over = loss(over, target)
    assert loss_under > loss_over


def test_demand_mlp_forward_shape():
    model = DemandMLP(n_features=5, activation="gelu")
    x = torch.randn(8, 5)
    out = model(x)
    assert out.shape == (8,)


@pytest.mark.parametrize("activation", ["relu", "gelu", "swish"])
def test_train_mlp_reduces_loss(synthetic_df, activation):
    X, y, _ = build_features(synthetic_df)
    split = make_split(X, y)
    model, scaler, history = train_mlp(split, activation=activation, epochs=60)
    assert len(history) == 60
    # La loss del final del entrenamiento debe ser menor que al inicio.
    assert history[-1] < history[0]
    preds = predict_mlp(model, scaler, split.X_test)
    assert preds.shape == split.y_test.shape
    assert np.all(np.isfinite(preds))


def test_run_comparison_covers_all_models(synthetic_df):
    results = run_comparison(synthetic_df)
    expected = {"ridge_baseline", "random_forest", "mlp_relu", "mlp_gelu", "mlp_swish"}
    assert expected == set(results["models"].keys())
    for name, payload in results["models"].items():
        metrics = payload["metrics"]
        assert set(metrics.keys()) == {"mae", "rmse", "r2"}
        assert metrics["mae"] >= 0
        assert metrics["rmse"] >= 0


def test_persist_and_read_metrics_duckdb(tmp_path: Path, synthetic_df):
    results = run_comparison(synthetic_df)
    db_path = tmp_path / "metrics.duckdb"
    persist_metrics_duckdb(results, db_path=db_path, run_label="unit_test")
    df = read_metrics_duckdb(db_path=db_path)
    assert len(df) == len(results["models"])
    assert set(df["model_name"]) == set(results["models"].keys())
    assert (df["run_label"] == "unit_test").all()


def test_persist_metrics_duckdb_overwrites_same_run_label(tmp_path: Path, synthetic_df):
    results = run_comparison(synthetic_df)
    db_path = tmp_path / "metrics.duckdb"
    persist_metrics_duckdb(results, db_path=db_path, run_label="rerun")
    persist_metrics_duckdb(results, db_path=db_path, run_label="rerun")
    df = read_metrics_duckdb(db_path=db_path)
    # Un segundo run con el mismo label no debe duplicar filas.
    assert len(df) == len(results["models"])
