"""Polígonos reales de comunas de la Región Metropolitana (Santiago, Chile),
reemplazando los círculos sintéticos de la Fase 1 por geometría real y
permitiendo un clipping real de los puntos de demanda generados (un punto
solo es válido si cae *dentro* del polígono real de su comuna, no dentro de
un círculo aproximado).

No existe un shapefile oficial con licencia explícita y URL estable
verificada en este proyecto; se usa un mirror público, ampliamente
reutilizado en proyectos de datos abiertos chilenos, de los límites
comunales oficiales de la Región Metropolitana (información pública de
Chile, base cartográfica INE/SII). Ver `SOURCE_URL` y el disclaimer de datos
en el README.
"""

from __future__ import annotations

import unicodedata
import urllib.request
from pathlib import Path

import geopandas as gpd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_DIR = DATA_DIR / "raw"
FULL_DOWNLOAD_PATH = RAW_DIR / "_rm_comunas_full.geojson"
COMUNAS_SUBSET_PATH = RAW_DIR / "comunas_rm_subset.geojson"

SOURCE_URL = "https://raw.githubusercontent.com/caracena/chile-geojson/master/13.geojson"

# Nombres tal como los usa el resto del proyecto (`geo_generator.COMUNAS`).
TARGET_COMUNAS = ["Las Condes", "Providencia", "Santiago Centro", "Maipú", "San Bernardo"]
# El polígono oficial llama "Santiago" a lo que este proyecto, desde Fase 1,
# llama "Santiago Centro" (para distinguirlo de la Región Metropolitana).
COMUNA_ALIASES = {"Santiago Centro": "Santiago"}


def _normalize(name: str) -> str:
    """Minúsculas y sin tildes, para matchear nombres pese a diferencias de
    encoding/acentuación entre fuentes."""
    nfkd = unicodedata.normalize("NFKD", name)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()


def fetch_comunas_geojson(output_path: Path = FULL_DOWNLOAD_PATH, url: str = SOURCE_URL) -> Path:
    """Descarga el GeoJSON completo de las 52 comunas de la Región
    Metropolitana desde la fuente pública (requiere conexión a internet).
    Solo hace falta para *regenerar* `comunas_rm_subset.geojson`; el resto
    del proyecto usa `load_target_comunas`, que lee el subconjunto ya
    filtrado y comiteado en el repo -- así el pipeline principal corre
    offline."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, output_path)
    return output_path


def filter_target_comunas(full_geojson_path: Path, comuna_names: list[str] = TARGET_COMUNAS) -> gpd.GeoDataFrame:
    """Filtra el GeoJSON completo de la RM a solo las comunas objetivo,
    normalizando nombres y re-aplicando el alias "Santiago Centro"."""
    gdf = gpd.read_file(full_geojson_path)

    normalized_targets = {_normalize(COMUNA_ALIASES.get(n, n)) for n in comuna_names}
    gdf["_norm_name"] = gdf["Comuna"].apply(_normalize)
    subset = gdf[gdf["_norm_name"].isin(normalized_targets)].copy()

    found = set(subset["_norm_name"])
    missing = normalized_targets - found
    if missing:
        raise ValueError(f"No se encontraron estas comunas en la fuente: {missing}")

    subset = subset.drop(columns="_norm_name").rename(columns={"Comuna": "comuna"})[["comuna", "geometry"]]
    reverse_alias = {_normalize(v): k for k, v in COMUNA_ALIASES.items()}
    subset["comuna"] = subset["comuna"].apply(lambda n: reverse_alias.get(_normalize(n), n))
    return subset.reset_index(drop=True).to_crs("EPSG:4326")


def save_target_comunas_subset(subset: gpd.GeoDataFrame, path: Path = COMUNAS_SUBSET_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    subset.to_file(path, driver="GeoJSON")


def load_target_comunas(path: Path = COMUNAS_SUBSET_PATH) -> gpd.GeoDataFrame:
    """Carga el subconjunto de polígonos reales ya filtrado y comiteado en el
    repo (`data/raw/comunas_rm_subset.geojson`) -- no requiere red. Usar
    `fetch_comunas_geojson` + `filter_target_comunas` solo para regenerar ese
    archivo desde la fuente original."""
    return gpd.read_file(path)


def main() -> None:
    full_path = fetch_comunas_geojson()
    subset = filter_target_comunas(full_path)
    save_target_comunas_subset(subset)
    print(f"Comunas objetivo encontradas: {sorted(subset['comuna'].tolist())}")
    print(f"Guardado en {COMUNAS_SUBSET_PATH}")


if __name__ == "__main__":
    main()
