"""Station registry. Add a station: subclass Station and list it here."""

from typing import Dict

from .base import ChartSpec, Station
from .skierg import SkiErg

STATIONS: Dict[str, Station] = {s.key: s for s in (SkiErg(),)}

__all__ = ["STATIONS", "Station", "ChartSpec"]
