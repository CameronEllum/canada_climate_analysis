from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import typer

logger = logging.getLogger(__name__)


class AggregateMode(str, Enum):
    monthly = "monthly"
    seasonally = "seasonally"
    yearly = "yearly"


@dataclass
class ProcessingConfig:
    """Holds processed and validated execution settings."""

    locations: tuple[str, ...]
    radius: float
    start_year: int
    end_year: Optional[int]
    trend: bool
    median: bool
    show_anomaly: bool
    max_temp: bool
    min_temp: bool
    period: str
    percentiles: list[int]

    @classmethod
    def from_args(
        cls,
        location: list[str],
        radius: float,
        start_year: int,
        end_year: Optional[int],
        trend: bool,
        median: bool,
        show_anomaly: bool,
        max_temp: bool,
        min_temp: bool,
        mode: AggregateMode,
        percentiles: list[int],
    ) -> ProcessingConfig:
        """Constructs and validates the config from raw CLI arguments."""
        if not location:
            logger.error("--location is required unless using --cache-report.")
            raise typer.Exit(code=1)

        return cls(
            locations=tuple(location),
            radius=radius,
            start_year=start_year,
            end_year=end_year,
            trend=trend,
            median=median,
            show_anomaly=show_anomaly,
            max_temp=max_temp,
            min_temp=min_temp,
            period=mode.value,
            percentiles=percentiles,
        )
