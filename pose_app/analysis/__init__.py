"""Offline technique analysis of recorded video (Hyrox stations).

Pipeline:
    video -> pose per frame -> athlete track -> smoothed body metrics
          -> rep segmentation -> per-rep metrics -> rules -> report / annotated video
"""
