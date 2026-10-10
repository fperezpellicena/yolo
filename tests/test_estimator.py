"""The rtmlib (RTMPose) backend, against a stand-in rtmlib.

Run: python tests/test_estimator.py (or pytest). Needs neither rtmlib nor a model:
a fake `rtmlib` module answers like BodyWithFeet's detector and pose model,
from the synthetic SkiErg athlete, with nine extra Halpe-26 points per person.
"""
import os, sys, tempfile, types
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from synthetic_skierg import FakeEstimator, KINDS, write_video


class StubDetector:
    def __init__(self, source):
        self.source, self.score_thr, self.people = source, 0.7, []

    def __call__(self, frame):
        self.people = self.source(frame)
        return np.array([p.box for p in self.people]) if self.people else np.array([])


class StubPose:
    def __init__(self, det):
        self.det, self.calls = det, 0

    def __call__(self, frame, bboxes=()):
        self.calls += 1
        assert len(bboxes) == len(self.det.people) > 0, "pose model run without detections"
        kp = np.array([p.keypoints for p in self.det.people], dtype=np.float32)
        sc = np.array([p.scores for p in self.det.people], dtype=np.float32)
        n = len(kp)   # Halpe-26: nine more points after COCO-17, here obviously wrong
        kp = np.concatenate([kp, np.full((n, 9, 2), -999.0, np.float32)], axis=1)
        sc = np.concatenate([sc, np.full((n, 9), 0.99, np.float32)], axis=1)
        return kp, sc


def install_stub_rtmlib(source):
    """A fake `rtmlib` whose BodyWithFeet answers from `source(frame)` -> [Person]."""
    made = {}

    class BodyWithFeet:
        def __init__(self, mode="balanced", backend="onnxruntime", device="cpu", **_):
            made.update(mode=mode, backend=backend, device=device)
            self.det_model = StubDetector(source)
            self.pose_model = StubPose(self.det_model)
            made["solution"] = self

    sys.modules["rtmlib"] = types.SimpleNamespace(BodyWithFeet=BodyWithFeet)
    return made


def test_model_names_and_devices():
    from pose_app.rtm_estimator import is_rtm_model, rtm_device
    assert all(is_rtm_model(m) for m in ("rtmpose", "rtmpose-s", "RTMPose-M", "rtmpose-x"))
    assert not any(is_rtm_model(m) for m in ("yolo11m-pose.pt", "rtmpose-q", "fake"))
    assert rtm_device("0") == "cuda:0" and rtm_device("cuda:1") == "cuda:1"
    assert rtm_device("cpu") == "cpu" and rtm_device("mps") == "mps"
    assert rtm_device(None) in ("cpu", "cuda")


def test_detect_keeps_coco17_and_sorts_largest_first():
    fake = FakeEstimator()
    made = install_stub_rtmlib(fake.detect)
    from pose_app.estimator import load_estimator
    est = load_estimator("rtmpose-x", 0.4, 960, "0", 0.5)
    assert made["mode"] == "performance" and made["device"] == "cuda:0"
    assert est.det_model.score_thr == 0.4

    people = est.detect(np.zeros((720, 1280, 3), np.uint8))
    assert len(people) == 2
    assert all(p.keypoints.shape == (17, 2) and p.scores.shape == (17,) for p in people)
    assert not (people[0].keypoints == -999).any()
    assert people[0].area > people[1].area
    assert any(v is not None for v in people[0].angles.values())

    people, img = est.estimate(np.zeros((720, 1280, 3), np.uint8))
    assert img.shape == (720, 1280, 3) and img.any()      # skeletons drawn on a copy


def test_no_detection_means_no_person():
    install_stub_rtmlib(lambda frame: [])
    from pose_app.estimator import load_estimator
    est = load_estimator("rtmpose", 0.4, 960, "cpu", 0.5)
    assert est.detect(np.zeros((64, 64, 3), np.uint8)) == []
    assert est.pose_model.calls == 0


def test_run_analysis_through_the_rtmlib_backend():
    """The whole analysis runs on it unchanged, and gives what the YOLO path gives."""
    from pose_app.analysis.api import AnalysisOptions, run_analysis
    with tempfile.TemporaryDirectory() as d:
        video = os.path.join(d, "ski.mp4")
        write_video(video)
        opts = dict(video=False, report=False, csv=False)
        yolo = run_analysis(video, os.path.join(d, "yolo"),
                            AnalysisOptions(model="fake", **opts), FakeEstimator())
        install_stub_rtmlib(FakeEstimator().detect)
        rtm = run_analysis(video, os.path.join(d, "rtm"), AnalysisOptions(model="rtmpose-m", **opts))
        assert len(rtm.result.reps) == len(yolo.result.reps) == len(KINDS)
        assert np.allclose(rtm.result.body.keypoints, yolo.result.body.keypoints, equal_nan=True)
        with np.load(rtm.files["pose_cache"]) as cache:     # the model is part of the cache
            assert str(cache["model"]) == "rtmpose-m"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: ok")
