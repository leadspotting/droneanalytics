import pytest
from pyproj import Transformer

from coordinate_converter import CoordinateConverter


@pytest.mark.parametrize('lat,lon,epsg', [(32.08, 34.78, 32636), (-33.86, 151.21, 32756)])
def test_projection_matches_hemisphere_epsg(lat, lon, epsg):
    converter = CoordinateConverter([lat], [lon])
    expected = Transformer.from_crs(4326, epsg, always_xy=True).transform(lon, lat)
    actual = converter.to_utm(lat, lon)
    assert actual == pytest.approx(expected, abs=0.001)
    assert converter.to_wgs84(*actual) == pytest.approx((lat, lon), abs=1e-7)


@pytest.mark.parametrize('lats,lons', [([32], [34, 35]), ([float('nan')], [34]), ([91], [34])])
def test_invalid_coordinates_are_rejected(lats, lons):
    with pytest.raises(ValueError):
        CoordinateConverter(lats, lons)
