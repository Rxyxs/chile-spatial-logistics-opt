from src.spatial.comunas import TARGET_COMUNAS, load_target_comunas


def test_load_target_comunas_returns_all_five():
    gdf = load_target_comunas()
    assert sorted(gdf["comuna"].tolist()) == sorted(TARGET_COMUNAS)


def test_all_polygons_are_valid_and_nonzero_area():
    gdf = load_target_comunas()
    assert gdf.geometry.is_valid.all()
    # UTM 19S (metros): area en un CRS proyectado, no en grados de un CRS
    # geografico como EPSG:4326 (donde 'area' no tiene una unidad fisica real).
    projected = gdf.to_crs("EPSG:32719")
    assert (projected.geometry.area > 0).all()


def test_crs_is_wgs84():
    gdf = load_target_comunas()
    assert gdf.crs is not None
    assert gdf.crs.to_epsg() == 4326


def test_polygons_are_within_plausible_santiago_bounds():
    """Sanity check on the real downloaded geometry -- every comuna's
    centroid should fall within the greater Santiago metro area, not
    somewhere else in Chile (would indicate a wrong file/parsing bug)."""
    gdf = load_target_comunas()
    centroids = gdf.to_crs("EPSG:32719").geometry.centroid.to_crs("EPSG:4326")
    assert centroids.y.between(-33.8, -33.2).all()  # latitud
    assert centroids.x.between(-71.0, -70.3).all()  # longitud
