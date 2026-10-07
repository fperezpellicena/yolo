"""
Webcam pose estimation with a joint-angle readout.

Package layout (one responsibility per module):

    cli.py               command-line parsing and program start-up
    app.py               PoseApp: main loop and keyboard controls
    config.py            Settings: every runtime option in one place
    cameras.py           webcam discovery and opening
    camera_selection.py  interactive "which camera?" prompt
    smoothing.py         AngleSmoother: jitter reduction
    fps.py               FpsMeter
    recorder.py          Recorder: fixed-size video output
    ui/                  everything that draws pixels or owns the window

Start with `python -m pose_app.live` (or `python -m pose_app`).
"""
