import unittest.mock as mock

import polars as pl
import pytest
import requests

from climate_cache import ClimateCache
from msc_client import MSCClient


@pytest.fixture
def mock_cache():
    return mock.MagicMock(spec=ClimateCache)


@pytest.fixture
def client(mock_cache):
    return MSCClient(mock_cache)


def test_get_coordinates_successful(client):
    """Verify coordinate fetching with mock geocoder."""
    with mock.patch("msc_client.Nominatim") as mock_nom:
        instance = mock_nom.return_value
        instance.geocode.return_value = mock.MagicMock(
            latitude=45.5, longitude=-73.5
        )

        coords = client.get_coordinates("Montreal, QC")

        assert coords == (45.5, -73.5)
        instance.geocode.assert_called_once()


def test_find_stations_api_mock(client, mock_cache):
    """Test station search with mocked API response."""
    mock_response = mock.MagicMock()
    mock_response.json.return_value = {
        "features": [
            {
                "properties": {
                    "CLIMATE_IDENTIFIER": "S1",
                    "STATION_NAME": "Station 1",
                },
                "geometry": {"coordinates": [-73.5, 45.5]},
            }
        ]
    }
    client.session.get = mock.MagicMock(return_value=mock_response)

    # Radius covers Montreal
    stations = client.find_stations_near(45.5, -73.5, 10.0)

    assert len(stations) == 1
    assert stations["id"][0] == "S1"
    assert stations["name"][0] == "Station 1"
    mock_cache.save_stations.assert_called_once()


def test_fetch_daily_data_orchestration(client, mock_cache):
    """Test that fetch_daily_data calls gap detection and then fetches blocks."""
    sid = "S1"

    # Define gaps
    mock_cache.get_missing_blocks.return_value = [
        (sid, "2020-01-01", "2020-01-31")
    ]

    # Mock API for block fetch
    mock_response = mock.MagicMock()
    mock_response.json.return_value = {
        "features": [
            {
                "properties": {
                    "CLIMATE_IDENTIFIER": sid,
                    "LOCAL_DATE": "2020-01-01",
                    "MEAN_TEMPERATURE": 5.5,
                    "TOTAL_PRECIPITATION": 10.0,
                }
            }
        ]
    }
    client.session.get = mock.MagicMock(return_value=mock_response)

    # Define final retrieval
    mock_cache.get_daily_data.return_value = pl.DataFrame(
        [
            {
                "station_id": sid,
                "date": "2020-01-01",
                "temp_mean": 5.5,
                "precip_total": 10.0,
            }
        ]
    )

    result = client.fetch_daily_data([sid], 2020, 2020)

    # Verify blocks were detected
    mock_cache.get_missing_blocks.assert_called_with([sid], 2020, 2020)
    # Verify API called for the gap
    client.session.get.assert_called()
    # Verify save_daily_request called instead of separate save/log
    mock_cache.save_daily_request.assert_called_once()
    # Verify final data retrieved
    assert len(result) == 1


def test_api_failure_handling(client, mock_cache):
    """Ensure client handles API errors without crashing."""
    client.session.get = mock.MagicMock(
        side_effect=requests.RequestException("API Down")
    )

    # Should not raise exception
    client._fetch_and_cache_block("S1", "2020-01-01", "2020-01-31")

    # save_daily_request should NOT be called if API failed
    mock_cache.save_daily_request.assert_not_called()
