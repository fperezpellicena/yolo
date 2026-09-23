"""Temporal smoothing of angle readings."""

import numpy as np

from .skeleton import Angles


class AngleSmoother:
    """Exponential moving average per angle so the readout does not jitter.

    alpha = 1.0 disables smoothing; smaller values are smoother but lag more.
    """

    def __init__(self, alpha: float = 0.4):
        self.alpha = float(np.clip(alpha, 0.01, 1.0))
        self._state: dict = {}

    def __call__(self, angles: Angles) -> Angles:
        if self.alpha >= 1.0:
            return angles
        out: Angles = {}
        for label, value in angles.items():
            if value is None:
                self._state.pop(label, None)
                out[label] = None
                continue
            previous = self._state.get(label)
            smoothed = value if previous is None else \
                self.alpha * value + (1.0 - self.alpha) * previous
            self._state[label] = smoothed
            out[label] = smoothed
        return out

    def reset(self) -> None:
        self._state.clear()
