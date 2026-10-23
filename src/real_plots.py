"""Figures for the real-data comparison (`src.real_santiago` calls `draw_all`)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PLOTS = Path(__file__).resolve().parents[1] / "outputs" / "plots"
SIM = "#9aa5b1"
REAL = "#c0392b"
INK = "#2b2b2b"


def _save(fig, name: str) -> Path:
    PLOTS.mkdir(parents=True, exist_ok=True)
    path = PLOTS / name
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {path.name}")
    return path


def demand_maps(comunas, panels: list[tuple[str, object, object]]) -> Path:
    """One map per demand model: demand per H3 cell and the three depots it picks."""
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 5.6))
    vmax = max(float(cells["total_demand"].max()) for _, cells, _ in panels)
    for ax, (title, cells, depots) in zip(axes, panels):
        comunas.boundary.plot(ax=ax, color="#555555", linewidth=0.7)
        cells.plot(ax=ax, column="total_demand", cmap="YlOrRd", vmin=0, vmax=vmax, linewidth=0)
        ax.scatter(depots["centroid_lon"], depots["centroid_lat"], marker="*", s=260, color="#1f4e79",
                   edgecolor="white", linewidth=1.2, zorder=5, label="depots (top-3 demand cells)")
        for r in comunas.itertuples():
            p = r.geometry.representative_point()
            ax.annotate(r.comuna, (p.x, p.y), fontsize=7.5, color=INK, ha="center")
        ax.set_title(title, fontsize=10.5)
        ax.set_axis_off()
    axes[0].legend(loc="lower left", fontsize=8, frameon=False)
    fig.suptitle("Same 1,025 units of demand, three ways to place them: where the depots go depends on the data",
                 fontsize=12, y=1.02)
    return _save(fig, "real_demand_maps.png")


def circuity_histogram(c: dict) -> Path:
    edges = np.array(c["histogram"]["edges"])
    counts = np.array(c["histogram"]["counts"], dtype=float)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.bar(edges[:-1], counts / counts.sum(), width=np.diff(edges), align="edge", color=REAL, alpha=0.8,
           edgecolor="white", label=f"Real (OSRM), {c['pairs']:,} pairs; median {c['all']['50']}")
    lo, hi = c["simulated_intra_range"]
    ax.axvspan(lo, hi, color=SIM, alpha=0.35, label=f"Simulated, same comuna {lo}-{hi}")
    lo2, hi2 = c["simulated_inter_range"]
    ax.axvspan(lo2, hi2, ymin=0, ymax=0.5, color="#1f4e79", alpha=0.18, label=f"Simulated, across comunas {lo2}-{hi2}")
    ax.set_xlabel("Road distance / straight-line distance")
    ax.set_ylabel("Share of location pairs")
    ax.set_xlim(1.0, 3.0)
    ax.set_title(f"Real roads are longer and more variable than the simulated factor: "
                 f"{c['share_above_simulated_maximum']:.0%} of pairs exceed its maximum,\n"
                 f"and the detour is larger inside a comuna (median {c['intra_comuna']['50']}) "
                 f"than across comunas ({c['inter_comuna']['50']})", fontsize=10.5)
    ax.legend(frameon=False, fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, "real_circuity.png")


def routing_km(r: dict) -> Path:
    s = r["synthetic_demand_simulated_roads"]
    p = r["real_demand_simulated_roads"]
    q = r["real_demand_real_roads"]
    rows = [("Synthetic demand,\nsimulated roads", s["total_km"], s["total_km_range"], SIM),
            ("Real demand,\nsimulated roads (plan)", p["total_km"], p["total_km_range"], SIM),
            ("Same plan driven\non real roads", p["driven_on_real_roads_km"], p["driven_on_real_roads_km_range"], REAL),
            ("Real demand,\nplanned on real roads", q["total_km"], q["total_km_range"], "#1f4e79")]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    vals = [x[1] for x in rows]
    err = [[v - x[2][0] for v, x in zip(vals, rows)], [x[2][1] - v for v, x in zip(vals, rows)]]
    bars = ax.bar([x[0] for x in rows], vals, color=[x[3] for x in rows], width=0.6,
                  yerr=err, capsize=5, ecolor=INK)
    for b, x in zip(bars, rows):
        ax.text(b.get_x() + b.get_width() / 2, x[2][1] + 12, f"{x[1]:,.0f} km", ha="center", fontsize=10)
    gap = p["driven_on_real_roads_km"] / p["total_km"] - 1
    saved = 1 - q["total_km"] / p["driven_on_real_roads_km"]
    ax.set_ylabel("Total route length, km (median, range of runs)")
    ax.set_title(f"A plan made on simulated distances is {gap:.0%} longer once driven;\n"
                 f"planning on the real distances cuts that by {saved:.0%}", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, "real_routing_km.png")


def forecast_cv(f: dict) -> Path:
    cv = f["cross_validation"]
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    x = np.arange(2)
    for k, (name, color) in enumerate((("random_folds", SIM), ("spatial_blocks", REAL))):
        vals = [cv[name]["ridge"], cv[name]["random_forest"]]
        b = ax.bar(x + (k - 0.5) * 0.36, vals, 0.34, color=color,
                   label="random 5-fold" if k == 0 else f"whole neighbourhoods held out ({cv['n_blocks']} blocks)")
        for bb, v in zip(b, vals):
            ax.text(bb.get_x() + bb.get_width() / 2, v + (0.02 if v >= 0 else -0.07), f"{v:.2f}", ha="center", fontsize=9.5)
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["Ridge", "Random Forest"])
    ax.set_ylabel("R² predicting cell population from location")
    ax.set_title("Location 'predicts' real demand only by copying the neighbours", fontsize=11)
    ax.legend(frameon=False, fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, "real_forecast_cv.png")
