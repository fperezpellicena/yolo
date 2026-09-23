"""
Webcam pose estimation with a joint-angle readout.

Package layout (one responsibility per module):

    cli.py               command-line parsing and program start-up
    app.py               PoseApp: main loop and keyboard controls
    config.py            Settings: every runtime option in one place
    cameras.py           webcam discovery and opening
    camera_selection.py  interactive "which camera?" prompt
    estimator.py         PoseEstimator: YOLO inference -> Person records
    person.py            Person: one detected body
    skeleton.py          COCO-17 keypoints and the joint-angle definitions
    geometry.py          angle maths
    smoothing.py         AngleSmoother: jitter reduction
    fps.py               FpsMeter
    recorder.py          Recorder: fixed-size video output
    ui/                  everything that draws pixels or owns the window
"""

import os

# Silence OpenCV's backend chatter while probing camera indices, and the
# per-frame Ultralytics logging. Must be set before cv2/ultralytics load,
# which is why it lives in the package __init__.
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
os.environ.setdefault("YOLO_VERBOSE", "False")

__version__ = "2.0.0"
