# chile-spatial-logistics-opt

Sistema de Inteligencia Geoespacial para optimizar la ubicación de **Dark
Stores** y las rutas de **entrega de última milla** en la Región
Metropolitana de Santiago, Chile.

**Fase 1** implementa el pipeline completo: generación sintética de demanda →
indexación espacial H3 → identificación de celdas candidatas a Dark Store →
optimización de rutas (VRP con capacidad y ventanas de tiempo) → dashboard
interactivo.

## 🛠️ Stack

- Python 3.11+
- GeoPandas / Shapely — análisis geoespacial
- H3-py (Uber) — indexación hexagonal, resolución 8
- Google OR-Tools — Vehicle Routing Problem (VRP) con capacidad y ventanas de tiempo
- Streamlit + Folium — dashboard de mapas interactivos
- Pytest — validación de restricciones geométricas y de ruteo

## 📁 Estructura

```
chile-spatial-logistics-opt/
├── data/
│   ├── raw/                     # insumos externos (vacío en Fase 1)
│   └── processed/                # datasets generados (CSV / GeoJSON)
├── src/
│   ├── spatial/
│   │   └── geo_generator.py      # demanda sintética + indexación H3
│   ├── optimization/
│   │   └── vrp_solver.py         # solucionador VRP (OR-Tools)
│   └── app/
│       └── streamlit_map.py      # dashboard Streamlit + Folium
├── tests/
│   ├── test_geo_generator.py
│   └── test_vrp_solver.py
├── scripts/
│   └── auto_push.py              # sincronización con GitHub (add/commit/push)
├── requirements.txt
└── README.md
```

## 🚀 Instalación

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

> **Nota (Windows):** `geopandas`/`pyogrio` dependen de binarios GDAL. Si
> `pip install` falla al compilar, usa Conda como alternativa:
> `conda install -c conda-forge geopandas h3-py ortools`.

## ▶️ Uso

### 1. Generar el dataset de demanda + indexación H3

```powershell
python -m src.spatial.geo_generator
```

Genera pedidos sintéticos en Las Condes, Providencia, Santiago Centro, Maipú
y San Bernardo, los indexa en celdas H3 (resolución 8) y guarda:

- `data/processed/demand_points.csv` — pedidos individuales
- `data/processed/demand_points.geojson`
- `data/processed/h3_demand_agg.geojson` — demanda agregada por celda H3

### 2. Resolver el VRP de forma independiente

```powershell
python -m src.optimization.vrp_solver
```

Toma la celda H3 de mayor demanda como Dark Store (depósito) y calcula rutas
óptimas para 4 furgones, minimizando distancia y balanceando carga.

### 3. Levantar el dashboard interactivo

```powershell
streamlit run src/app/streamlit_map.py
```

Permite ajustar el volumen de pedidos, número de furgones, capacidad por
vehículo y candidatos a Dark Store, visualizando en un mapa de Santiago las
celdas H3 por nivel de demanda y las rutas optimizadas (un color por
vehículo).

### 4. Ejecutar los tests

```powershell
pytest
```

Cubre: límites geográficos de los puntos generados, resolución H3 correcta,
consistencia de la agregación por celda, factibilidad del VRP, cumplimiento
de capacidades por vehículo y que cada punto de demanda se visite exactamente
una vez.

### 5. Sincronizar con GitHub

```powershell
python scripts/auto_push.py -m "mensaje de commit"
```

Ejecuta `git add -A`, pide confirmación interactiva y luego hace `commit` +
`push` a la rama indicada (`--branch`, por defecto `main`). Requiere que el
repositorio ya tenga un remoto `origin` configurado.

## 🗺️ Diseño del modelo

- **Demanda sintética**: puntos generados de forma uniforme dentro de un
  radio (km) alrededor del centro de cada comuna, con una cantidad de
  demanda aleatoria por pedido.
- **Indexación H3 (resolución 8)**: cada pedido se asigna a una celda
  hexagonal (~0.46 km² de área promedio), unidad natural para agregar
  demanda urbana y elegir ubicaciones de Dark Store.
- **Selección de Dark Store**: heurística inicial — las celdas H3 con mayor
  demanda agregada son las candidatas (`select_dark_store_candidates`).
- **VRP**: modelo de OR-Tools con dimensión de distancia (para el costo y el
  balanceo de carga vía `SetGlobalSpanCostCoefficient`), dimensión de
  capacidad por vehículo y dimensión de tiempo con ventanas de entrega.

## 🔭 Próximas fases (fuera del alcance de Fase 1)

- Reemplazar los círculos sintéticos por polígonos reales de comunas (INE /
  OpenStreetMap) y clipping real de puntos.
- Matriz de distancias por red vial real (OSRM / Valhalla) en vez de
  distancia haversine.
- Optimización multi-depósito (facility location) para elegir el número y
  ubicación óptima de Dark Stores simultáneamente con el ruteo.
- Persistencia de escenarios y comparación de métricas entre corridas.
