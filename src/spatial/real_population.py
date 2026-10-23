"""Real demand: resident population per H3 cell (resolution 8, the grid this repo uses).

Two public sources, combined:

- **Kontur Population 2023-11-01** (HDX, CC BY 4.0): population per H3
  resolution-8 hexagon, disaggregated from global built-up and census inputs.
  It gives the *distribution inside* each comuna.
- **Censo 2017 (INE)**: the official resident count per comuna. Kontur's own
  comuna totals disagree with it badly for these five comunas (Santiago Centro
  -62%, Maipu +83%), so each comuna's cells are rescaled to the census total.

Demand is then spread over the cells in proportion to population, keeping the
synthetic module's total (so the two scenarios differ only in *where* the
demand is, not how much there is).
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import h3
import numpy as np
import pandas as pd

from src.spatial.geo_generator import RAW_DIR, _h3_cell_to_polygon

KONTUR_PATH = RAW_DIR / "kontur" / "kontur_population_CL_20231101.gpkg"

# INE, Censo de Poblacion y Vivienda 2017, poblacion efectivamente censada por comuna.
CENSO_2017 = {
    "Santiago Centro": 404_495,
    "Providencia": 142_079,
    "Las Condes": 294_838,
    "Maipú": 521_627,
    "San Bernardo": 301_313,
}


def load_kontur_cells(comunas: gpd.GeoDataFrame, path: Path = KONTUR_PATH) -> gpd.GeoDataFrame:
    """Kontur hexagons whose centroid falls inside one of the target comunas."""
    k = gpd.read_file(path).to_crs(32719)
    cent = k[["h3", "population"]].copy()
    cent = gpd.GeoDataFrame(cent, geometry=k.geometry.centroid, crs=32719)
    joined = gpd.sjoin(cent, comunas[["comuna", "geometry"]].to_crs(32719), predicate="within")
    return cells_frame(joined["h3"].tolist(), joined["population"].astype(float).tolist(),
                       joined["comuna"].tolist())


def cells_frame(cells: list[str], population: list[float], comuna: list[str]) -> gpd.GeoDataFrame:
    df = pd.DataFrame({"h3_cell": cells, "kontur_population": population, "comuna": comuna})
    ll = df["h3_cell"].apply(h3.cell_to_latlng)
    df["centroid_lat"] = ll.apply(lambda c: c[0])
    df["centroid_lon"] = ll.apply(lambda c: c[1])
    df["geometry"] = df["h3_cell"].apply(_h3_cell_to_polygon)
    return gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")


def calibrate_to_census(cells: gpd.GeoDataFrame, census: dict[str, int] = CENSO_2017) -> gpd.GeoDataFrame:
    """Rescale each comuna's cells so they sum to its census total."""
    out = cells.copy()
    totals = out.groupby("comuna")["kontur_population"].transform("sum")
    target = out["comuna"].map(census).astype(float)
    out["population"] = out["kontur_population"] * target / totals
    return out


def comuna_comparison(cells: gpd.GeoDataFrame, census: dict[str, int] = CENSO_2017) -> pd.DataFrame:
    k = cells.groupby("comuna")["kontur_population"].agg(cells="size", kontur="sum")
    k["censo_2017"] = k.index.map(census)
    k["kontur_vs_censo_pct"] = 100 * (k["kontur"] / k["censo_2017"] - 1)
    return k.reset_index()


def demand_from_population(cells: gpd.GeoDataFrame, total_units: int, column: str = "population",
                           drop_zero: bool = True) -> gpd.GeoDataFrame:
    """Integer demand per cell, proportional to `column`, summing to `total_units`
    (largest-remainder rounding). Cells left with no demand are dropped."""
    share = cells[column].to_numpy(dtype=float)
    share = share / share.sum()
    raw = share * total_units
    units = np.floor(raw).astype(int)
    remainder = total_units - units.sum()
    units[np.argsort(-(raw - units))[:remainder]] += 1
    out = cells.copy()
    out["total_demand"] = units
    if drop_zero:
        out = out[out["total_demand"] > 0]
    return out.reset_index(drop=True)
