"""Rep detection on hand-made signals; no video or model needed.
Run: python tests/test_signal.py (or pytest).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from pose_app.analysis.signal import find_cycles

FPS = 30
T = np.arange(0, 10.5, 1 / FPS)            # peaks (catches) at 0, 2, 4, 6, 8 and 10 s
Y = np.cos(np.pi * T)


def _cycles(y):
    return find_cycles(y, T, lo=-0.5, hi=0.5, min_s=0.7, max_s=5.0)


def test_one_cycle_per_stroke():
    cycles = _cycles(Y)
    assert [round(T[c.start]) for c in cycles] == [0, 2, 4, 6, 8]
    assert round(T[cycles[-1].end]) == 10        # the clip ends after the final catch


def test_missing_data_at_the_end_keeps_the_final_rep():
    y = Y.copy()
    y[-3:] = np.nan                              # e.g. a missed detection in the last frames
    assert len(_cycles(y)) == 5


def test_missing_data_inside_a_rep_breaks_it():
    y = Y.copy()
    y[(T > 4.8) & (T < 5.2)] = np.nan            # around the trough of the third stroke
    assert [round(T[c.start]) for c in _cycles(y)] == [0, 2, 6, 8]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: ok")
