"""Frame-rate measurement."""

import time
from typing import Optional


class FpsMeter:
    """Smoothed frames-per-second from successive tick() calls."""

    def __init__(self, smoothing: float = 0.9):
        self.smoothing = smoothing
        self.value = 0.0
        self._last: Optional[float] = None

    def tick(self) -> float:
        now = time.perf_counter()
        if self._last is not None:
            delta = now - self._last
            if delta > 0:
                instant = 1.0 / delta
                self.value = instant if self.value == 0.0 else \
                    self.smoothing * self.value + (1.0 - self.smoothing) * instant
        self._last = now
        return self.value

    def restart(self) -> None:
        """Forget the last timestamp (e.g. after a pause) without losing the value."""
        self._last = None
