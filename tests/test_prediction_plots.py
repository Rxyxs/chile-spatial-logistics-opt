from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.prediction.demand_forecast import run_comparison
from src.prediction.plots import plot_activation_curves, plot_actual_vs_predicted, plot_model_comparison


@pytest.fixture(scope="module")
def results():
    rng = np.random.default_rng(1)
    n = 50
    df = pd.DataFrame(
        {
            "comuna": rng.choice(["Las Condes", "Providencia"], size=n),
            "centroid_lat": rng.uniform(-33.6, -33.4, size=n),
            "centroid_lon": rng.uniform(-70.7, -70.5, size=n),
            "total_demand": rng.integers(1, 25, size=n).astype(float),
        }
    )
    return run_comparison(df)


def test_plot_model_comparison_creates_file(tmp_path: Path, results):
    out = tmp_path / "comparison.png"
    path = plot_model_comparison(results, out_path=out)
    assert path.exists()
    assert path.stat().st_size > 0


def test_plot_activation_curves_creates_file(tmp_path: Path, results):
    out = tmp_path / "convergence.png"
    path = plot_activation_curves(results, out_path=out)
    assert path.exists()
    assert path.stat().st_size > 0


def test_plot_actual_vs_predicted_creates_file(tmp_path: Path, results):
    out = tmp_path / "scatter.png"
    path = plot_actual_vs_predicted(results, out_path=out)
    assert path.exists()
    assert path.stat().st_size > 0
