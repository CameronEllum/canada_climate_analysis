"""
Unit tests for climate trend calculation and anomaly computation.

These tests verify that:
1. Linear trend is calculated correctly from yearly averages
2. Anomalies are computed as (actual - trend)
3. Edge cases are handled properly
"""

import polars as pl
import pytest

from report_generator import calculate_trendline
from report_plots import _add_anomaly_columns


class TestTrendlineCalculation:
    """Test the linear trendline calculation function."""

    def test_simple_increasing_trend(self):
        """Test trendline with simple increasing values."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [1.0, 2.0, 3.0, 4.0, 5.0]  # Perfect linear: y = x

        trend = calculate_trendline(x, y)

        assert trend is not None
        assert len(trend) == 5
        # Should be very close to the original values
        for i, val in enumerate(trend):
            assert abs(val - y[i]) < 0.01

    def test_horizontal_trend(self):
        """Test trendline with constant values."""
        x = [2000.0, 2001.0, 2002.0, 2003.0]
        y = [5.0, 5.0, 5.0, 5.0]  # Flat line

        trend = calculate_trendline(x, y)

        assert trend is not None
        for val in trend:
            assert abs(val - 5.0) < 0.01

    def test_negative_slope(self):
        """Test trendline with decreasing values."""
        x = [1.0, 2.0, 3.0, 4.0]
        y = [10.0, 8.0, 6.0, 4.0]  # y = 12 - 2x

        trend = calculate_trendline(x, y)

        assert trend is not None
        # Verify slope is negative
        assert trend[0] > trend[-1]
        # Check specific values
        assert abs(trend[0] - 10.0) < 0.01
        assert abs(trend[-1] - 4.0) < 0.01

    def test_with_noise(self):
        """Test trendline with noisy data around a trend."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [1.1, 1.9, 3.2, 3.8, 5.1]  # Approximately y = x

        trend = calculate_trendline(x, y)

        assert trend is not None
        # Trend should be close to y = x
        for i, val in enumerate(trend):
            assert abs(val - x[i]) < 0.5

    def test_insufficient_data(self):
        """Test that single data point returns None."""
        x = [1.0]
        y = [5.0]

        trend = calculate_trendline(x, y)

        assert trend is None

    def test_empty_data(self):
        """Test that empty data returns None."""
        x = []
        y = []

        trend = calculate_trendline(x, y)

        assert trend is None

    def test_with_none_values(self):
        """Test trendline calculation with None values in data."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [1.0, None, 3.0, None, 5.0]  # Some None values

        trend = calculate_trendline(x, y)

        # Should still calculate trend using valid points
        assert trend is not None
        assert len(trend) == 5
        # Trend should be close to y = x for valid points
        assert abs(trend[0] - 1.0) < 0.5
        assert abs(trend[4] - 5.0) < 0.5

    def test_all_none_values(self):
        """Test that all None values returns None."""
        x = [1.0, 2.0, 3.0]
        y = [None, None, None]

        trend = calculate_trendline(x, y)

        assert trend is None


class TestAnomalyCalculation:
    """Test the anomaly calculation function."""

    def test_anomaly_with_perfect_trend(self):
        """Test anomaly calculation when data matches trend perfectly."""
        stats = pl.DataFrame(
            {
                "year": [2000, 2001, 2002, 2003],
                "avg": [10.0, 11.0, 12.0, 13.0],  # Perfect linear trend
            }
        )

        result = _add_anomaly_columns(stats)

        assert "anomaly" in result.columns
        assert "trend" in result.columns
        # Anomalies should be very close to zero
        for anom in result["anomaly"].to_list():
            assert abs(anom) < 0.01

    def test_anomaly_with_deviation(self):
        """Test anomaly calculation with actual deviations."""
        stats = pl.DataFrame(
            {
                "year": [2000, 2001, 2002, 2003],
                "avg": [10.0, 12.0, 11.0, 13.0],  # Deviates from perfect trend
            }
        )

        result = _add_anomaly_columns(stats)

        assert "anomaly" in result.columns
        # Calculate expected trend manually: y ≈ 9 + 1x (approximately)
        # Year 2001 (12.0) should be above trend
        # Year 2002 (11.0) should be below trend
        anomalies = result["anomaly"].to_list()
        assert anomalies[1] > 0  # 2001 above trend
        assert anomalies[2] < 0  # 2002 below trend

    def test_anomaly_columns_present(self):
        """Test that all required columns are added."""
        stats = pl.DataFrame(
            {
                "year": [2020, 2021, 2022],
                "avg": [-5.0, -4.5, -4.0],
            }
        )

        result = _add_anomaly_columns(stats)

        assert "trend" in result.columns
        assert "anomaly" in result.columns
        assert len(result) == 3

    def test_anomaly_with_single_point(self):
        """Test anomaly calculation with insufficient data for trend."""
        stats = pl.DataFrame(
            {
                "year": [2020],
                "avg": [10.0],
            }
        )

        result = _add_anomaly_columns(stats)

        # Should fall back to long-term mean
        assert "anomaly" in result.columns
        assert result["anomaly"][0] == 0.0  # Deviation from mean is 0

    def test_realistic_temperature_data(self):
        """Test with realistic temperature data showing warming trend."""
        # Simulating January temperatures with warming trend
        stats = pl.DataFrame(
            {
                "year": [1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020],
                "avg": [-7.0, -6.8, -6.5, -6.0, -5.5, -5.0, -4.5, -4.0],
            }
        )

        result = _add_anomaly_columns(stats)

        # Verify trend exists and is increasing (warming)
        trends = result["trend"].to_list()
        assert trends[0] < trends[-1]  # Warming trend

        # Verify anomalies are calculated
        assert "anomaly" in result.columns

    def test_realistic_precipitation_data(self):
        """Test with realistic precipitation data."""
        # Simulating July precipitation with slight increasing trend
        stats = pl.DataFrame(
            {
                "year": [1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020],
                "avg": [45.0, 48.0, 46.0, 50.0, 49.0, 52.0, 51.0, 54.0],
            }
        )

        result = _add_anomaly_columns(stats)

        # Verify all required columns
        assert "trend" in result.columns
        assert "anomaly" in result.columns
        assert len(result) == 8


class TestTrendMethodology:
    """Test that trend calculation follows the documented methodology."""

    def test_trend_uses_yearly_averages(self):
        """
        Verify that trend is calculated from yearly averages for each month.

        Methodology:
        1. For each month (e.g., January), we have yearly averages
        2. Linear regression is fit through these yearly averages
        3. Trend value for each year is the regression line value
        """
        # January data: yearly averages across all stations
        stats = pl.DataFrame(
            {
                "year": [2018, 2019, 2020, 2021, 2022],
                "avg": [-5.0, -4.8, -4.6, -4.4, -4.2],  # Warming trend
            }
        )

        result = _add_anomaly_columns(stats)

        # Trend should be linear through these points
        trends = result["trend"].to_list()

        # Verify linearity: differences should be approximately equal
        diffs = [trends[i + 1] - trends[i] for i in range(len(trends) - 1)]
        avg_diff = sum(diffs) / len(diffs)
        for diff in diffs:
            assert abs(diff - avg_diff) < 0.01  # Linear within tolerance

    def test_anomaly_is_actual_minus_trend(self):
        """
        Verify that anomaly = actual value - trend value.

        This is the core definition of climate anomaly.
        """
        stats = pl.DataFrame(
            {
                "year": [2020, 2021, 2022],
                "avg": [10.0, 11.0, 12.0],
            }
        )

        result = _add_anomaly_columns(stats)

        # Manually verify: anomaly = avg - trend
        for i in range(len(result)):
            expected_anomaly = result["avg"][i] - result["trend"][i]
            actual_anomaly = result["anomaly"][i]
            assert abs(expected_anomaly - actual_anomaly) < 0.001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
