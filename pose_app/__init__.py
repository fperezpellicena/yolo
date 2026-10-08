"""
Pose estimation (Ultralytics YOLO) for three uses, one subpackage each, plus
the PM5 recorder that feeds the force analysis:

    live/        webcam app with a joint-angle readout     python -m pose_app.live
    analysis/    offline technique analysis of a video     python -m pose_app.analysis
    worker/      runs analyses queued by the web app       python -m pose_app.worker
    pm5/         records / decodes a Concept2 PM5 log      python -m pose_app.pm5

They share the modules at this level, and depend only on them (the worker
also on analysis/, analysis on pm5/ to read PM5 logs), never on each other:

    estimator.py         PoseEstimator: YOLO inference -> Person records
    person.py            Person: one detected body
    skeleton.py          COCO-17 keypoints and the joint-angle definitions
    geometry.py          angle maths
    drawing/             fonts, colours and text helpers for OpenCV frames

`python -m pose_app` starts the webcam app.
"""

import os

# Silence OpenCV's backend chatter while probing camera indices, and the
# per-frame Ultralytics logging. Must be set before cv2/ultralytics load,
# which is why it lives in the package __init__.
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
os.environ.setdefault("YOLO_VERBOSE", "False")

__version__ = "2.0.0"
