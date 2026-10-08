"""Force analysis (SkiErg with a Concept2 PM5): from the cord tension to the joints.

    setup.py        athlete profile + calibration (cord exit, image scale)
    model.py        de Leva body segments, sized to the athlete
    kinematics.py   zero-lag filtering, keypoints -> metres, segment motion
    dynamics.py     Newton-Euler joint moments, joint power, energy, force capacity
    sync.py         PM5 strokes <-> video strokes; curves onto frames by cord travel
    analysis.py     the whole chain + per-stroke f_* metrics and checks
    rules.py        force-based technique rules, fatigue checks, charts
    overlay.py      the force lines drawn on the annotated video
"""

from .analysis import FORCE_KEYS, ForceAnalysis, analyze_force, attach_force, sync_by_hands
from .rules import FORCE_CHARTS, force_drift_rules, force_rules
from .setup import SEXES, AthleteProfile, Calibration, ForceSetup
from .sync import StrokeSync

__all__ = ["FORCE_KEYS", "ForceAnalysis", "analyze_force", "attach_force", "sync_by_hands",
           "FORCE_CHARTS", "force_drift_rules", "force_rules", "SEXES", "AthleteProfile",
           "Calibration", "ForceSetup", "StrokeSync"]
