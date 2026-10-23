"""The same pipeline on real demand and real roads: `python -m src.real_santiago`.

The notebook's results rest on two simulated inputs: demand drawn uniformly
inside each comuna (45 orders per comuna, demand independent of location) and
road distances simulated as straight-line distance times a random circuity
factor. This study swaps each for a real one and measures what changes:

1. Demand: resident population per H3 cell (Kontur, rescaled to Censo 2017
   comuna totals), with the same 1,025 total units as the synthetic run.
2. Depots: the top-demand cells under each demand model.
3. Roads: OSRM driving distances on OpenStreetMap; the real circuity against
   the simulated ranges, and the cost of planning routes on the simulated
   matrix and then driving them on real roads.
4. Forecasting: whether location now predicts demand (it could not on the
   synthetic data, R2 ~ 0), under a random split and under a spatial one.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import h3
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

from src.optimization.multi_depot_vrp import (
    build_multi_depot_instance, solve_multi_depot_vrp, summarize_by_depot,
)
from src import real_plots
from src.prediction.demand_forecast import build_features, run_comparison
from src.spatial.comunas import load_target_comunas
from src.spatial.geo_generator import (
    RAW_DIR, aggregate_by_h3, assign_h3_index, generate_demand_within_polygons, select_dark_store_candidates,
)
from src.spatial.network_distance import (
    INTER_ZONE_CIRCUITY_RANGE, INTRA_ZONE_CIRCUITY_RANGE, vectorized_haversine_matrix,
)
from src.spatial.osrm import circuity, road_distance_matrix
from src.spatial.real_population import (
    calibrate_to_census, comuna_comparison, demand_from_population, load_kontur_cells,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "reports" / "real_santiago.json"
OSRM_CACHE = RAW_DIR / "osrm" / "real_demand_matrix.npy"

N_DEPOTS = 3
VEHICLES_PER_DEPOT = [3, 3, 3]
VEHICLE_CAPACITY = 150
SEED = 42
SOLVE_SECONDS = 60
REPEATS = 3   # OR-Tools stops on a time limit, so its answer varies from run to run


def synthetic_cells(comunas) -> pd.DataFrame:
    """The notebook's synthetic demand, unchanged."""
    return aggregate_by_h3(assign_h3_index(generate_demand_within_polygons(comunas, n_per_comuna=45, seed=SEED)))


def split_depots(cells):
    cand = select_dark_store_candidates(cells, n_stores=N_DEPOTS)
    depots = list(zip(cand["centroid_lat"], cand["centroid_lon"]))
    rest = cells[~cells["h3_cell"].isin(cand["h3_cell"])].reset_index(drop=True)
    return cand, depots, rest


def route_km(solution, matrix: np.ndarray) -> float:
    """Length of a solution's routes measured on another distance matrix."""
    total = 0.0
    for r in solution.routes:
        stops = list(r.stops) if hasattr(r, "stops") else list(r.nodes)
        total += sum(matrix[a, b] for a, b in zip(stops[:-1], stops[1:]))
    return float(total)


def solve(rest, depots, labels, matrix=None) -> dict:
    inst = build_multi_depot_instance(rest, depots, vehicles_per_depot=VEHICLES_PER_DEPOT,
                                      vehicle_capacity=VEHICLE_CAPACITY, seed=SEED)
    if matrix is not None:
        inst.distance_matrix_km = matrix
    sol = solve_multi_depot_vrp(inst, time_limit_seconds=SOLVE_SECONDS)
    if sol is None:
        return {"feasible": False, "instance": inst}
    return {"feasible": True, "instance": inst, "solution": sol,
            "total_km": float(sol.total_distance_km), "vehicles_used": len(sol.routes),
            "dropped": len(sol.dropped),
            "by_depot": summarize_by_depot(sol, inst.depot_indices, labels)}


def comuna_shares(cells) -> dict:
    s = cells.groupby("comuna")["total_demand"].sum()
    return {k: round(float(v / s.sum()), 4) for k, v in s.items()}


def spatial_cv(df: pd.DataFrame, parent_res: int = 6, folds: int = 5) -> dict:
    """R2 when whole neighbourhoods (H3 resolution-6 parents, ~36 km2) are held out,
    against a random split of the same cells."""
    X, y, _ = build_features(df)
    groups = df["h3_cell"].apply(lambda c: h3.cell_to_parent(c, parent_res)).to_numpy()
    rng = np.random.default_rng(SEED)
    random_groups = rng.integers(0, folds, len(df))
    out = {}
    for name, g in (("random_folds", random_groups), ("spatial_blocks", groups)):
        n_folds = min(folds, len(np.unique(g)))
        scores = {"ridge": [], "random_forest": []}
        for tr, te in GroupKFold(n_splits=n_folds).split(X, y, g):
            ridge = Ridge(alpha=1.0).fit(X[tr], y[tr])
            rf = RandomForestRegressor(n_estimators=200, max_depth=6, min_samples_leaf=2,
                                       random_state=SEED).fit(X[tr], y[tr])
            scores["ridge"].append(r2_score(y[te], ridge.predict(X[te])))
            scores["random_forest"].append(r2_score(y[te], rf.predict(X[te])))
        out[name] = {k: round(float(np.mean(v)), 4) for k, v in scores.items()}
    out["n_blocks"] = int(len(np.unique(groups)))
    return out


def main() -> dict:
    start = time.perf_counter()
    comunas = load_target_comunas()
    report: dict = {}

    synth = synthetic_cells(comunas)
    total_units = int(synth["total_demand"].sum())
    kontur = load_kontur_cells(comunas)
    report["population_sources"] = comuna_comparison(kontur).to_dict(orient="records")
    real_all = calibrate_to_census(kontur)
    real = demand_from_population(real_all, total_units)
    raw = demand_from_population(kontur, total_units, column="kontur_population")
    print(f"synthetic: {len(synth)} cells; real: {len(real)} cells with demand of {len(real_all)}; units {total_units}")

    report["demand"] = {
        "total_units": total_units,
        "synthetic": {"cells": int(len(synth)), "comuna_share": comuna_shares(synth)},
        "real_census_calibrated": {"cells": int(len(real)), "comuna_share": comuna_shares(real)},
        "real_kontur_raw": {"cells": int(len(raw)), "comuna_share": comuna_shares(raw)},
    }

    depots_of = {}
    for name, cells in (("synthetic", synth), ("real_kontur_raw", raw), ("real_census_calibrated", real)):
        cand, depots, rest = split_depots(cells)
        depots_of[name] = (cand, depots, rest)
        report.setdefault("depots", {})[name] = [
            {"h3_cell": r.h3_cell, "comuna": r.comuna, "demand": int(r.total_demand)} for r in cand.itertuples()]
    print("depots:", {k: [d["comuna"] for d in v] for k, v in report["depots"].items()})

    # Real road distances for the real-demand instance (depots first, as the solver expects).
    cand, depots, rest = depots_of["real_census_calibrated"]
    locations = depots + list(zip(rest["centroid_lat"], rest["centroid_lon"]))
    road = road_distance_matrix(locations, cache=OSRM_CACHE)
    straight = vectorized_haversine_matrix(locations)
    zones = ["__depot__"] * len(depots) + rest["comuna"].tolist()
    same = np.array([[a == b for b in zones] for a in zones])
    c_all = circuity(road, straight)
    mask = straight >= 0.5
    c_intra, c_inter = road[mask & same] / straight[mask & same], road[mask & ~same] / straight[mask & ~same]
    q = lambda a: {str(p): round(float(np.percentile(a, p)), 3) for p in (10, 50, 90)}
    report["circuity"] = {
        "pairs": int(c_all.size), "all": q(c_all), "intra_comuna": q(c_intra), "inter_comuna": q(c_inter),
        "simulated_intra_range": list(INTRA_ZONE_CIRCUITY_RANGE), "simulated_inter_range": list(INTER_ZONE_CIRCUITY_RANGE),
        "share_below_simulated_minimum": round(float(np.mean(c_all < INTRA_ZONE_CIRCUITY_RANGE[0])), 4),
        "share_above_simulated_maximum": round(float(np.mean(c_all > INTER_ZONE_CIRCUITY_RANGE[1])), 4),
        "asymmetry_median_pct": round(float(np.median(np.abs(road - road.T)[mask] / road[mask]) * 100), 2),
    }
    print("circuity", report["circuity"]["all"], report["circuity"]["intra_comuna"], report["circuity"]["inter_comuna"])
    edges = np.arange(1.0, 3.01, 0.05)
    report["circuity"]["histogram"] = {"edges": edges.round(2).tolist(),
                                       "counts": np.histogram(np.clip(c_all, 1.0, 3.0), edges)[0].tolist()}

    labels = lambda cand: [f"{r.comuna} ({int(r.total_demand)})" for r in cand.itertuples()]
    s_cand, s_dep, s_rest = depots_of["synthetic"]
    runs = {"synthetic_demand_simulated_roads": [], "real_demand_simulated_roads": [], "real_demand_real_roads": []}
    for _ in range(REPEATS):
        runs["synthetic_demand_simulated_roads"].append(solve(s_rest, s_dep, labels(s_cand)))
        planned = solve(rest, depots, labels(cand))
        if planned["feasible"]:
            planned["driven_on_real_roads_km"] = route_km(planned["solution"], road)
        runs["real_demand_simulated_roads"].append(planned)
        runs["real_demand_real_roads"].append(solve(rest, depots, labels(cand), matrix=road))
    report["routing"] = {}
    for k, reps in runs.items():
        keep = [{kk: vv for kk, vv in r.items() if kk not in ("instance", "solution")} for r in reps]
        summary = {"repeats": keep}
        for metric in ("total_km", "driven_on_real_roads_km", "vehicles_used", "dropped"):
            vals = [r[metric] for r in keep if r.get(metric) is not None]
            if vals:
                summary[metric] = float(np.median(vals))
                summary[f"{metric}_range"] = [float(min(vals)), float(max(vals))]
        report["routing"][k] = summary
        print(k, {m: summary.get(m) for m in ("total_km", "total_km_range", "driven_on_real_roads_km",
                                                "driven_on_real_roads_km_range", "dropped")})

    fc_all = real_all[["h3_cell", "comuna", "centroid_lat", "centroid_lon"]].copy()
    fc_all["total_demand"] = real_all["population"].to_numpy()
    comp = run_comparison(fc_all)
    report["forecast"] = {
        "target": "census-calibrated population per H3 cell",
        "cells": int(len(fc_all)),
        "random_75_25_split": {k: {m: round(x, 4) for m, x in v["metrics"].items()} for k, v in comp["models"].items()},
        "cross_validation": spatial_cv(fc_all),
    }
    print("forecast", report["forecast"]["cross_validation"])

    real_plots.demand_maps(comunas, [
        ("Synthetic (uniform inside each comuna)", synth, depots_of["synthetic"][0]),
        ("Kontur population, as published", raw, depots_of["real_kontur_raw"][0]),
        ("Kontur rescaled to Censo 2017", real, depots_of["real_census_calibrated"][0]),
    ])
    real_plots.circuity_histogram(report["circuity"])
    real_plots.routing_km(report["routing"])
    real_plots.forecast_cv(report["forecast"])

    report["elapsed_seconds"] = round(time.perf_counter() - start, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(f"wrote {OUT} in {report['elapsed_seconds']} s")
    return report


if __name__ == "__main__":
    main()
