"""Offline technique analysis of recorded video (Hyrox stations).

Pipeline:
    video -> pose per frame -> athlete track -> smoothed body metrics
          -> rep segmentation -> per-rep metrics -> rules -> report / annotated video
Machine data (telemetry/) and, on the SkiErg with a PM5 log, the force
analysis (force/) add their per-rep metrics before the rules run.
"""
