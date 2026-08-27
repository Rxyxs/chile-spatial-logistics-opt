[ 🇺🇸 English ] | [ 🇨🇱 [Español](README.es.md) ]

# 🚚 Chile Spatial Logistics Optimizer

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat&logo=python&logoColor=white)
![GeoPandas](https://img.shields.io/badge/GeoPandas-real%20polygons-3776AB?style=flat)
![OR-Tools](https://img.shields.io/badge/OR--Tools-Multi--Depot%20VRPTW-4285F4?style=flat)
![Folium](https://img.shields.io/badge/Folium-interactive%20maps-77B829?style=flat)
![Pydeck](https://img.shields.io/badge/Pydeck-3D%20arcs-4B32C3?style=flat)
![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-2%20notebooks-F37626?style=flat&logo=jupyter&logoColor=white)
![Tests](https://img.shields.io/badge/tests-40%20passing-brightgreen?style=flat&logo=pytest&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

Geospatial intelligence system for **Dark Store** siting and **last-mile delivery** route optimization in Santiago, Chile's Metropolitan Region.

**Phase 1** built the full single-depot pipeline: synthetic demand → H3 spatial indexing → Dark Store candidate selection → VRP with capacity and time windows → interactive dashboard.

**Phase 2** (this update) replaces three of Phase 1's own documented simplifications with the real thing: real comuna polygons instead of circles, a simulated road-network distance matrix instead of straight-line distance, and a genuine multi-depot VRPTW solved jointly instead of one depot at a time — plus standalone HTML map export (Folium + Pydeck) and a companion evaluation notebook.

## 1. Architecture

```mermaid
flowchart TB
    A["comunas.py<br/>real comuna polygons<br/>(public GeoJSON mirror)"] --> B["geo_generator.py<br/>demand clipped to real polygons<br/>(rejection sampling)"]
    B --> C["H3 spatial index (res 8)<br/>+ select_dark_store_candidates<br/>(N depot candidates)"]
    C --> D["network_distance.py<br/>vectorized Haversine<br/>+ simulated road-network matrix"]
    D --> E["multi_depot_vrp.py<br/>OR-Tools Multi-Depot VRPTW<br/>native starts/ends per vehicle"]
    E --> F["map_export.py<br/>Folium (2D) + Pydeck (3D arcs)<br/>standalone HTML"]
    E --> G["streamlit_map.py<br/>interactive dashboard<br/>(single-depot, Phase 1)"]
```

## 2. Design decisions

- **Real comuna polygons, not circles**: `src/spatial/comunas.py` loads real administrative boundaries for the 5 target comunas (Las Condes, Providencia, Santiago Centro, Maipú, San Bernardo) from a public GeoJSON mirror of Chile's official communal cartography (see §7, Data sources). `generate_demand_within_polygons` uses rejection sampling — draw a point inside the polygon's bounding box, keep it only if it's actually inside the real polygon — so every generated order is guaranteed to fall within its comuna's real boundary, unlike Phase 1's circle approximation (which could place a point in a neighboring comuna).
- **Simulated road-network distance, not a live OSRM server**: standing up real OSRM requires a `.osm.pbf` road-network extract (hundreds of MB) plus a preprocessing pipeline (`osrm-extract`/`osrm-contract`) — infrastructure this project doesn't need to demonstrate that the routing engine can consume *any* distance matrix, not just straight-line. Instead, `src/spatial/network_distance.py` applies a **circuity factor** (road distance ÷ straight-line distance, literature-typical range 1.15–1.35 within a zone, 1.25–1.55 crossing zones) on top of a vectorized Haversine matrix — explicitly disclosed as simulated, never claimed to be a real OSRM query, and built so road distance is never shorter than straight-line distance (a physical invariant, tested directly).
- **"Accelerated Haversine"**: the vectorized NumPy implementation replaces Phase 1's O(n²) pure-Python double loop, with a measured (not claimed) speedup — see §6.
- **True multi-depot VRPTW, not pre-clustering**: `src/optimization/multi_depot_vrp.py` uses OR-Tools' native support for multiple vehicle `starts`/`ends` (`RoutingIndexManager(n, num_vehicles, starts, ends)`), so the solver decides *jointly* which vehicle — from which depot — serves which stop, in a single optimization. This matters: pre-assigning each demand cell to its nearest depot before routing can leave one depot's fleet overloaded while a farther depot with spare capacity sits idle; solving jointly avoids that failure mode by construction.
- **Standalone map export, not just an in-app dashboard**: `src/spatial/map_export.py` renders routes colored by depot (not by vehicle) — the question multi-depot optimization exists to answer is about load balance *between depots*, so that's the visual grouping that matters — to a self-contained HTML file, openable without running Streamlit.

## 3. Project structure

```
chile-spatial-logistics-opt/
├── data/
│   ├── raw/
│   │   └── comunas_rm_subset.geojson     # real comuna polygons (committed, ~25KB)
│   └── processed/                         # generated datasets (CSV / GeoJSON)
├── src/
│   ├── spatial/
│   │   ├── comunas.py                     # real comuna polygon loading/fetching
│   │   ├── geo_generator.py               # synthetic demand (circle + real-polygon variants), H3 indexing
│   │   ├── network_distance.py            # vectorized Haversine + simulated road-network matrix
│   │   └── map_export.py                  # Folium + Pydeck standalone HTML export
│   ├── optimization/
│   │   ├── vrp_solver.py                  # single-depot VRPTW (Phase 1)
│   │   └── multi_depot_vrp.py             # multi-depot VRPTW (Phase 2)
│   └── app/
│       └── streamlit_map.py               # interactive dashboard (single-depot)
├── 02_MultiDepot_VRPTW_OSRM.ipynb          # executed, real outputs
├── outputs/
│   └── maps/                              # 2 committed example maps (Folium + Pydeck)
├── tests/
├── scripts/
│   └── auto_push.py                       # git add/commit/push helper
├── requirements.txt
├── README.md
└── README.es.md
```

## 4. Installation

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

> **Windows note:** `geopandas`/`pyogrio` depend on GDAL binaries. If `pip install` fails to build, use Conda instead: `conda install -c conda-forge geopandas h3-py ortools`.

## 5. Usage

### 1. Generate the demand dataset + H3 index

```powershell
python -m src.spatial.geo_generator
```

### 2. (Optional) Re-fetch the real comuna polygons

```powershell
python -m src.spatial.comunas
```

Only needed to regenerate `data/raw/comunas_rm_subset.geojson` from the original public source — the committed file is already there, so the pipeline runs fully offline without this step.

### 3. Solve a single-depot VRP (Phase 1)

```powershell
python -m src.optimization.vrp_solver
```

### 4. Run the Multi-Depot VRPTW notebook (Phase 2)

```powershell
jupyter nbconvert --to notebook --execute --inplace 02_MultiDepot_VRPTW_OSRM.ipynb
# or open it interactively:
jupyter notebook 02_MultiDepot_VRPTW_OSRM.ipynb
```

Generates demand within real comuna polygons, selects 3 depot candidates, builds the simulated road-network distance matrix, solves the joint multi-depot VRPTW, shows the per-depot load-balance chart, and exports both the Folium and Pydeck maps to `outputs/maps/`.

### 5. Launch the interactive dashboard (single-depot)

```powershell
streamlit run src/app/streamlit_map.py
```

### 6. Run the tests

```powershell
pytest
```

### 7. Sync with GitHub

```powershell
python scripts/auto_push.py -m "commit message"
```

## 6. Results

Every number below comes from an actual run of `02_MultiDepot_VRPTW_OSRM.ipynb` (seed 42) — nothing here is estimated.

| Metric | Value |
|---|---|
| Real comuna polygons loaded | 5 (Las Condes, Providencia, Santiago Centro, Maipú, San Bernardo) |
| Demand points generated (clipped to real polygons) | 225 |
| H3 cells with demand (resolution 8) | 173 |
| Total demand | 1,025 units |
| Depot candidates selected | 3 |
| Haversine speedup (vectorized vs. pure-Python loop, 173 locations) | **28.7x** (43.49ms → 1.51ms) |
| Multi-Depot VRPTW total distance (simulated road network) | 503.90 km |
| Vehicles used | 8 / 9 available |
| Unassigned demand cells | 0 |
| Test suite | 40/40 passing (`pytest`) |

**Honest note on depot selection**: 2 of the 3 top-demand H3 cells selected as depot candidates both fall within Santiago Centro (different specific locations, same comuna) — a real, un-forced result of picking the highest-demand cells rather than one-per-comuna by design. It's a realistic outcome for a dense urban core (multiple dark stores in the same comuna is a legitimate operating model), not a bug, and not smoothed over into an artificially "diverse" example.

## 7. Data sources

- **Real comuna polygons**: `data/raw/comunas_rm_subset.geojson`, filtered from a public GitHub mirror of the Metropolitan Region's official communal boundaries (ultimately Chile's INE/SII cartographic base): [caracena/chile-geojson](https://github.com/caracena/chile-geojson), file `13.geojson` (region 13 = Región Metropolitana). No explicit license is stated on that mirror; the underlying administrative-boundary data itself is Chilean public information. See `src/spatial/comunas.py` for the fetch/filter script used to regenerate this file from source.
- **Road-network distances**: simulated, not a live OSRM query — see §2. Disclosed explicitly, not presented as real routing-engine output.
- **Demand and orders**: fully synthetic (own generator, seeded, deterministic).

## 8. Testing

```powershell
pytest -v
```

40 tests: real-polygon integrity (validity, CRS, plausible bounds), point-in-real-polygon clipping correctness (every generated point verified via spatial join to be inside its assigned comuna, not just its bounding box), vectorized-vs-loop Haversine numerical equivalence, road-network-distance physical invariants (never shorter than straight-line, symmetric, deterministic given a seed, higher circuity for inter-zone trips on average), multi-depot instance validation (rejects nonzero depot demand, mismatched vehicle/depot counts, wrong-shaped distance matrices), multi-depot solution correctness (every route starts and ends at its own depot, nearby demand is served by the nearby depot, per-depot summaries sum to the solution total), and map-export smoke tests (both Folium and Pydeck HTML files are created with real content).

## 9. Future work

- Facility-location optimization for *where* to place depot candidates (not just which top-demand H3 cells to pick), jointly with the routing decision.
- Persist optimization scenarios and compare metrics across runs.
- A real OSRM instance (Docker + a Santiago `.osm.pbf` extract) as a drop-in replacement for the simulated road-network matrix, behind the same `distance_matrix_km` interface `multi_depot_vrp.py` already expects.
- Extend the multi-depot dashboard (Streamlit) to match the notebook's capabilities, not just the single-depot Phase 1 UI.

## License

MIT — see [LICENSE](LICENSE).

## Author

**Pablo Reyes** — [github.com/Rxyxs](https://github.com/Rxyxs)
