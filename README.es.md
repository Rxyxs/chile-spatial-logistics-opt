[ 🇺🇸 [Read in English](README.md) ] | [ 🇨🇱 Español ]

# 🚚 Chile Spatial Logistics Optimizer

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat&logo=python&logoColor=white)
![GeoPandas](https://img.shields.io/badge/GeoPandas-poligonos%20reales-3776AB?style=flat)
![OR-Tools](https://img.shields.io/badge/OR--Tools-Multi--Depot%20VRPTW-4285F4?style=flat)
![Folium](https://img.shields.io/badge/Folium-mapas%20interactivos-77B829?style=flat)
![Pydeck](https://img.shields.io/badge/Pydeck-arcos%203D-4B32C3?style=flat)
![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-2%20notebooks-F37626?style=flat&logo=jupyter&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-MLP%20%2B%20loss%20custom-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![DuckDB](https://img.shields.io/badge/DuckDB-metricas-FFF000?style=flat&logo=duckdb&logoColor=black)
![Tests](https://img.shields.io/badge/tests-55%20passing-brightgreen?style=flat&logo=pytest&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

Sistema de Inteligencia Geoespacial para optimizar la ubicación de **Dark Stores** y las rutas de **entrega de última milla** en la Región Metropolitana de Santiago, Chile.

**Fase 1** implementó el pipeline completo de un solo depósito: demanda sintética → indexación espacial H3 → selección de celdas candidatas a Dark Store → VRP con capacidad y ventanas de tiempo → dashboard interactivo.

**Fase 2** reemplaza tres de las simplificaciones que la propia Fase 1 documentaba como pendientes: polígonos reales de comuna en vez de círculos, una matriz de distancias por red vial simulada en vez de línea recta, y un VRPTW multi-depósito genuino resuelto conjuntamente en vez de un depósito a la vez — más exportación de mapas HTML standalone (Folium + Pydeck) y un notebook de evaluación complementario.

**Fase 3** (esta actualización) agrega un **módulo de predicción de demanda** como insumo aguas arriba de la decisión de ubicación/ruteo, no como reemplazo de la optimización: dada solo la comuna y las coordenadas del centroide de una celda H3 candidata (la información disponible para un sitio sin historial de pedidos aún), predice su demanda esperada. Se comparan tres enfoques — un baseline de regresión Ridge, un ensamble Random Forest, y un MLP en PyTorch entrenado con una loss custom asimétrica (penaliza más la subestimación que la sobreestimación, porque un stockout en un Dark Store cuesta más que la capacidad ociosa) — a través de tres funciones de activación (ReLU, GELU, Swish/SiLU). Las métricas se persisten en DuckDB y los gráficos comparativos quedan versionados en `outputs/plots/`.

## 0.1 Impacto de Negocio e Indicadores Clave (KPIs)

| Métrica | Resultado | Qué significa |
|---|---|---|
| VRPTW multi-depósito, celdas de demanda sin asignar | **0** / 173 | Cada celda H3 de demanda servida, asignación depósito-vehículo-parada optimizada conjuntamente |
| Utilización de flota | 8 / 9 vehículos usados | Distancia total de ruta simulada: 503,90 km |
| Speedup de la matriz de distancias Haversine | **28,7x** (43,49ms → 1,51ms) | NumPy vectorizado vs. loop puro en Python, medido no afirmado |
| Polígonos reales de comuna | 5 comunas, 225 puntos de demanda | Muestreo por rechazo para que cada punto caiga genuinamente dentro de su polígono real, no un bounding box |
| Suite de tests | 55/55 pasando | Creció desde los 13 tests de la Fase 1, cubre contención de polígonos, invariantes de distancia por red, asignación multi-depósito y predicción de demanda |
| Predicción de demanda, mejor modelo (MAE) | 2,87 (Ridge) | Predice demanda de celda H3 solo por ubicación, como insumo para rankear candidatos a Dark Store |

## 1. Arquitectura

```mermaid
flowchart TB
    A["comunas.py<br/>poligonos reales de comuna<br/>(mirror publico GeoJSON)"] --> B["geo_generator.py<br/>demanda clippeada a poligonos reales<br/>(rejection sampling)"]
    B --> C["Indice espacial H3 (res 8)<br/>+ select_dark_store_candidates<br/>(N depositos candidatos)"]
    C --> D["network_distance.py<br/>Haversine vectorizada<br/>+ matriz de red vial simulada"]
    D --> E["multi_depot_vrp.py<br/>OR-Tools Multi-Depot VRPTW<br/>starts/ends nativos por vehiculo"]
    E --> F["map_export.py<br/>Folium (2D) + Pydeck (arcos 3D)<br/>HTML standalone"]
    E --> G["streamlit_map.py<br/>dashboard interactivo<br/>(un deposito, Fase 1)"]
    C --> H["demand_forecast.py<br/>Ridge / Random Forest / PyTorch MLP<br/>predice demanda de celdas H3 sin muestrear"]
    H --> I["plots.py + DuckDB<br/>comparacion de modelos, curvas de convergencia,<br/>persistencia de metricas"]
    H -.->|informa ranking de candidatos| C
```

## 2. Decisiones de diseño

- **Polígonos reales de comuna, no círculos**: `src/spatial/comunas.py` carga los límites administrativos reales de las 5 comunas objetivo (Las Condes, Providencia, Santiago Centro, Maipú, San Bernardo) desde un mirror público en GitHub de la cartografía comunal oficial de la Región Metropolitana (ver §7, Fuentes de datos). `generate_demand_within_polygons` usa rejection sampling — sortea un punto dentro del bounding box del polígono, lo conserva solo si cae realmente dentro del polígono real — así que todo pedido generado está garantizado de caer dentro del límite real de su comuna, a diferencia de la aproximación circular de Fase 1 (que podía ubicar un punto en la comuna vecina).
- **Distancia por red vial simulada, no un servidor OSRM real**: levantar OSRM de verdad requiere un extracto `.osm.pbf` de la red vial (cientos de MB) más un preprocesamiento (`osrm-extract`/`osrm-contract`) — infraestructura que este proyecto no necesita para demostrar que el motor de ruteo puede consumir *cualquier* matriz de distancias, no solo línea recta. En su lugar, `src/spatial/network_distance.py` aplica un **factor de circuidad** (distancia por calles ÷ distancia en línea recta, rango típico de literatura 1,15–1,35 dentro de una zona, 1,25–1,55 cruzando zonas) sobre una matriz Haversine vectorizada — declarado explícitamente como simulado, nunca presentado como una consulta OSRM real, y construido para que la distancia por calles nunca sea menor que la distancia en línea recta (un invariante físico, verificado directamente con un test).
- **"Haversine acelerada"**: la implementación vectorizada con NumPy reemplaza el doble loop en Python puro O(n²) de Fase 1, con un speedup medido (no solo afirmado) — ver §6.
- **Multi-Depot VRPTW real, no pre-clustering**: `src/optimization/multi_depot_vrp.py` usa el soporte nativo de OR-Tools para múltiples `starts`/`ends` de vehículo (`RoutingIndexManager(n, num_vehicles, starts, ends)`), así que el solver decide *conjuntamente* qué vehículo — de qué depósito — atiende cada parada, en una sola optimización. Esto importa: pre-asignar cada celda de demanda a su depósito más cercano antes de rutear puede dejar la flota de un depósito sobrecargada mientras un depósito más lejano con capacidad libre queda ocioso; resolver conjuntamente evita esa falla por construcción.
- **Exportación de mapas standalone, no solo un dashboard en la app**: `src/spatial/map_export.py` colorea las rutas por depósito (no por vehículo) — la pregunta que motiva la optimización multi-depósito es sobre el balance de carga *entre depósitos*, así que esa es la agrupación visual que importa — a un archivo HTML autocontenido, abrible sin correr Streamlit.
- **Predicción de demanda como insumo de la ubicación, no como reemplazo del optimizador**: `src/prediction/demand_forecast.py` predice `total_demand` para una celda H3 solo por ubicación (comuna + centroid lat/lon) — excluyendo deliberadamente `order_count`, que tampoco existiría todavía para un sitio candidato genuinamente nuevo. La salida está pensada para alimentar `select_dark_store_candidates`, no para reemplazar el solver VRP.
- **Loss asimétrica para el modelo neuronal**: `AsymmetricStockoutLoss` pondera la subestimación 2,5x sobre la sobreestimación, porque subestimar la demanda esperada de un Dark Store arriesga un stockout (ventas perdidas, clientes insatisfechos), mientras que sobreestimar solo cuesta capacidad ociosa — una asimetría propia del dominio que una MSE plana ignora.
- **Tres activaciones comparadas, no asumidas**: ReLU, GELU y Swish (`nn.SiLU`) se entrenan con datos/semilla/arquitectura idénticos, así que las diferencias de convergencia y precisión en `outputs/plots/mlp_activation_convergence.png` reflejan solo la elección de activación.

## 3. Estructura del proyecto

```
chile-spatial-logistics-opt/
├── data/
│   ├── raw/
│   │   └── comunas_rm_subset.geojson     # poligonos reales de comuna (comiteado, ~25KB)
│   └── processed/                         # datasets generados (CSV / GeoJSON)
├── src/
│   ├── spatial/
│   │   ├── comunas.py                     # carga/descarga de poligonos reales de comuna
│   │   ├── geo_generator.py               # demanda sintetica (variante circulo + poligono real), indexacion H3
│   │   ├── network_distance.py            # Haversine vectorizada + matriz de red vial simulada
│   │   └── map_export.py                  # exportacion HTML standalone Folium + Pydeck
│   ├── optimization/
│   │   ├── vrp_solver.py                  # VRPTW de un deposito (Fase 1)
│   │   └── multi_depot_vrp.py             # VRPTW multi-deposito (Fase 2)
│   ├── prediction/
│   │   ├── demand_forecast.py             # Ridge / Random Forest / PyTorch MLP para prediccion de demanda (Fase 3)
│   │   └── plots.py                       # graficos de comparacion / convergencia / real-vs-predicho
│   └── app/
│       └── streamlit_map.py               # dashboard interactivo (un deposito)
├── 02_MultiDepot_VRPTW_OSRM.ipynb          # ejecutado, salidas reales
├── outputs/
│   ├── maps/                              # 2 mapas de ejemplo comiteados (Folium + Pydeck)
│   ├── plots/                             # 3 graficos de ejemplo comiteados (prediccion de demanda)
│   └── metrics/                           # base DuckDB de metricas de prediccion (local, no versionada)
├── tests/
├── scripts/
│   └── auto_push.py                       # helper de git add/commit/push
├── requirements.txt
├── README.md
└── README.es.md
```

## 4. Instalación

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

> **Nota (Windows):** `geopandas`/`pyogrio` dependen de binarios GDAL. Si `pip install` falla al compilar, usa Conda como alternativa: `conda install -c conda-forge geopandas h3-py ortools`.

## 5. Uso

### 1. Generar el dataset de demanda + índice H3

```powershell
python -m src.spatial.geo_generator
```

### 2. (Opcional) Re-descargar los polígonos reales de comuna

```powershell
python -m src.spatial.comunas
```

Solo hace falta para regenerar `data/raw/comunas_rm_subset.geojson` desde la fuente pública original — el archivo ya está comiteado, así que el pipeline corre completamente offline sin este paso.

### 3. Resolver un VRP de un solo depósito (Fase 1)

```powershell
python -m src.optimization.vrp_solver
```

### 4. Correr el notebook Multi-Depot VRPTW (Fase 2)

```powershell
jupyter nbconvert --to notebook --execute --inplace 02_MultiDepot_VRPTW_OSRM.ipynb
# o abrirlo interactivamente:
jupyter notebook 02_MultiDepot_VRPTW_OSRM.ipynb
```

Genera demanda dentro de polígonos reales de comuna, selecciona 3 depósitos candidatos, construye la matriz de distancias por red vial simulada, resuelve el VRPTW multi-depósito conjunto, muestra el gráfico de balance de carga por depósito, y exporta los mapas Folium y Pydeck a `outputs/maps/`.

### 5. Levantar el dashboard interactivo (un depósito)

```powershell
streamlit run src/app/streamlit_map.py
```

### 5b. Correr la comparación de predicción de demanda (Fase 3)

```powershell
python -m src.prediction.demand_forecast
```

Entrena Ridge, Random Forest y un MLP en PyTorch (ReLU/GELU/Swish) sobre la agregación de demanda por H3, imprime MAE/RMSE/R² por modelo, persiste las métricas en `outputs/metrics/demand_forecast_metrics.duckdb`, y regenera los 3 gráficos en `outputs/plots/`.

### 6. Ejecutar los tests

```powershell
pytest
```

### 7. Sincronizar con GitHub

```powershell
python scripts/auto_push.py -m "mensaje de commit"
```

## 6. Resultados

Todos los números de abajo vienen de una corrida real de `02_MultiDepot_VRPTW_OSRM.ipynb` (semilla 42) — nada acá es estimado.

| Métrica | Valor |
|---|---|
| Polígonos reales de comuna cargados | 5 (Las Condes, Providencia, Santiago Centro, Maipú, San Bernardo) |
| Puntos de demanda generados (clippeados a polígonos reales) | 225 |
| Celdas H3 con demanda (resolución 8) | 173 |
| Demanda total | 1.025 unidades |
| Depósitos candidatos seleccionados | 3 |
| Speedup Haversine (vectorizada vs. loop puro en Python, 173 ubicaciones) | **28,7x** (43,49ms → 1,51ms) |
| Distancia total del VRPTW multi-depósito (red vial simulada) | 503,90 km |
| Vehículos usados | 8 / 9 disponibles |
| Celdas de demanda sin asignar | 0 |
| Suite de tests | 55/55 pasando (`pytest`) |

### 6.1 Predicción de demanda (Fase 3) -- comparación de modelos

Todos los números de abajo vienen de una corrida real de `python -m src.prediction.demand_forecast` (semilla 42) sobre la agregación de demanda por H3 descrita arriba (158 celdas, split 75%/25% train/test).

| Modelo | MAE | RMSE | R² |
|---|---|---|---|
| Ridge baseline (interpretable) | 2,87 | 4,60 | -0,01 |
| Random Forest (ensamble) | 3,05 | 4,85 | -0,13 |
| MLP PyTorch -- ReLU | 2,99 | 4,55 | 0,01 |
| MLP PyTorch -- GELU | 2,98 | 4,54 | 0,01 |
| MLP PyTorch -- Swish (SiLU) | 2,97 | 4,54 | 0,01 |

![Comparación de modelos](outputs/plots/demand_model_comparison.png)
La version animada traza las curvas de loss epoca a epoca, con una etiqueta flotante que sigue la loss actual de cada activacion.

![Convergencia por activación animada](outputs/plots/mlp_activation_convergence_animated.gif)
![Convergencia por activación](outputs/plots/mlp_activation_convergence.png)
![Real vs. predicho](outputs/plots/demand_actual_vs_predicted.png)

**Nota honesta sobre la precisión de la predicción**: un R² cercano a cero en los cinco modelos es el resultado correcto y sin maquillar acá — `total_demand` en el generador sintético se sortea independientemente de la ubicación (`rng.randint(min_demand, max_demand)` por pedido, sin correlación con comuna ni coordenadas por construcción), así que hay poca señal espacial real que aprender solo desde la ubicación. La comparación igual es significativa como **demostración de metodología** — tres familias de modelo genuinamente distintas, una loss asimétrica propia del dominio, y una comparación honesta de activaciones — que mostraría una brecha de precisión real sobre datos de demanda con estructura espacial genuina (estacionalidad, correlatos de ingreso/población a nivel comuna, cercanía a transporte), algo que este generador sintético no modela.

**Nota honesta sobre la selección de depósitos**: 2 de las 3 celdas H3 de mayor demanda seleccionadas como depósitos candidatos caen dentro de Santiago Centro (ubicaciones específicas distintas, misma comuna) — un resultado real y no forzado de elegir las celdas de mayor demanda en vez de una por comuna por diseño. Es un resultado realista para un centro urbano denso (varios dark stores en la misma comuna es un modelo operativo legítimo), no un bug, y no se suavizó hacia un ejemplo artificialmente "diverso".

## 7. Fuentes de datos

- **Polígonos reales de comuna**: `data/raw/comunas_rm_subset.geojson`, filtrado de un mirror público en GitHub de los límites comunales oficiales de la Región Metropolitana (en última instancia, la base cartográfica INE/SII de Chile): [caracena/chile-geojson](https://github.com/caracena/chile-geojson), archivo `13.geojson` (región 13 = Región Metropolitana). Ese mirror no declara una licencia explícita; el dato subyacente de límites administrativos es información pública chilena. Ver `src/spatial/comunas.py` para el script de descarga/filtrado usado para regenerar este archivo desde la fuente.
- **Distancias por red vial**: simuladas, no una consulta OSRM real — ver §2. Declarado explícitamente, no presentado como salida real de un motor de ruteo.
- **Demanda y pedidos**: completamente sintéticos (generador propio, con semilla, determinístico).

## 8. Testing

```powershell
pytest -v
```

55 tests: integridad de los polígonos reales (validez, CRS, límites plausibles), corrección del clipping punto-en-polígono-real (cada punto generado verificado vía spatial join de que efectivamente cae dentro de su comuna asignada, no solo dentro de su bounding box), equivalencia numérica Haversine vectorizada vs. loop, invariantes físicos de la distancia por red vial (nunca menor que la línea recta, simétrica, determinística dada una semilla, mayor circuidad promedio en viajes inter-zona), validación de la instancia multi-depósito (rechaza demanda de depósito distinta de cero, conteos de vehículos/depósitos no coincidentes, matrices de distancia con forma incorrecta), corrección de la solución multi-depósito (cada ruta empieza y termina en su propio depósito, la demanda cercana es atendida por el depósito cercano, los resúmenes por depósito suman el total de la solución), smoke tests de exportación de mapas (los archivos HTML de Folium y Pydeck se crean con contenido real), corrección de features/métricas de predicción de demanda (sin fuga de `order_count`, sanidad de MAE/RMSE/R² sobre una predicción perfecta), que la loss asimétrica efectivamente penalice más la subestimación que la sobreestimación, que la loss de entrenamiento del MLP baje en las 3 activaciones, persistencia de métricas en DuckDB (incluyendo sobreescritura idempotente al re-correr el mismo `run_label`), y smoke tests de generación de los 3 gráficos de predicción de demanda.

## 9. Próximos pasos

- Optimización de facility-location para *dónde* ubicar los depósitos candidatos (no solo qué celdas H3 de mayor demanda elegir), conjuntamente con la decisión de ruteo.
- Alimentar `select_dark_store_candidates` directamente con la demanda predicha por `demand_forecast.py`, para poder rankear celdas H3 sin muestrear como candidatas a depósito sin esperar historial real de pedidos.
- Datos de demanda con estructura espacial genuina (ingreso/población a nivel comuna, cercanía a transporte, estacionalidad) para darle a la comparación de modelos una brecha de precisión real que demostrar, no solo una de metodología.
- Persistencia de escenarios de optimización y comparación de métricas entre corridas.
- Una instancia real de OSRM (Docker + un extracto `.osm.pbf` de Santiago) como reemplazo directo de la matriz de red vial simulada, detrás de la misma interfaz `distance_matrix_km` que `multi_depot_vrp.py` ya espera.
- Extender el dashboard multi-depósito (Streamlit) para igualar las capacidades del notebook, no solo la UI de un depósito de Fase 1.

## Licencia

MIT — ver [LICENSE](LICENSE).

## Autor

**Pablo Reyes** — [github.com/Rxyxs](https://github.com/Rxyxs)
