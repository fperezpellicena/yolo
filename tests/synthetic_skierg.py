"""Synthetic SkiErg athlete (side view, facing image-right) for pipeline tests."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, cv2
from pose_app.person import Person
from pose_app.skeleton import KP
from pose_app.analysis.render import EDGES

FPS, W, H = 30, 960, 720
rng = np.random.default_rng(0)

def interp(th, keys):  # keys: list of (phase, value); smooth piecewise cosine
    for (p0, v0), (p1, v1) in zip(keys, keys[1:]):
        if p0 <= th <= p1:
            u = (th - p0) / (p1 - p0); u = (1 - np.cos(np.pi * u)) / 2
            return v0 + (v1 - v0) * u
    return keys[-1][1]

def pose(th, kind):
    d = np.radians
    if kind == "squat":
        shin = interp(th, [(0,5),(0.4,45),(1,5)]); thigh = interp(th, [(0,5),(0.4,55),(1,5)])
        trunk = interp(th, [(0,5),(0.4,30),(1,5)])
    else:
        shin = interp(th, [(0,5),(0.4,18),(1,5)]); thigh = interp(th, [(0,5),(0.4,35),(1,5)])
        trunk = interp(th, [(0,5),(0.4,60),(1,5)])
    alpha = interp(th, [(0,20),(0.4,185),(1,20)])
    if kind == "arms":
        beta = interp(th, [(0,15),(0.12,80),(0.3,60),(0.4,15),(1,15)])
    else:
        beta = interp(th, [(0,15),(0.2,18),(0.32,60),(0.4,15),(1,15)])
    A = np.array([420.0, 640.0])
    K = A + 150 * np.array([np.sin(d(shin)), -np.cos(d(shin))])
    Hp = K + 160 * np.array([-np.sin(d(thigh)), -np.cos(d(thigh))])
    S = Hp + 200 * np.array([np.sin(d(trunk)), -np.cos(d(trunk))])
    E = S + 110 * np.array([np.sin(d(alpha)), -np.cos(d(alpha))])
    Wr = E + 100 * np.array([np.sin(d(alpha + beta)), -np.cos(d(alpha + beta))])
    head = S + 55 * np.array([np.sin(d(trunk)), -np.cos(d(trunk))])
    kp = np.zeros((17, 2)); sc = np.full(17, 0.9)
    near = dict(shoulder=S, elbow=E, wrist=Wr, hip=Hp, knee=K, ankle=A)
    for j, p in near.items():
        kp[KP[f"left_{j}"]] = p; kp[KP[f"right_{j}"]] = p + [6, -2]; sc[KP[f"right_{j}"]] = 0.6
    kp[KP["left_ear"]] = head; kp[KP["right_ear"]] = head + [4, 0]; sc[KP["right_ear"]] = 0.5
    kp[KP["nose"]] = head + [22, 6]; kp[KP["left_eye"]] = head + [16, -2]; kp[KP["right_eye"]] = head + [18, -2]
    return kp, sc

def build():
    kinds = ["good"] * 8 + ["arms"] * 6 + ["squat"] * 6
    frames = []
    for n, kind in enumerate(kinds):
        period = 1.9 + 0.03 * n          # fatigue: slowing rate
        nf = int(period * FPS)
        for i in range(nf):
            frames.append(pose(i / nf, kind))
    frames += [pose(0.0, "good")] * 10  # finish tall
    return frames, kinds

FRAMES, KINDS = build()

def write_video(path):
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for kp, sc in FRAMES:
        img = np.full((H, W, 3), 60, np.uint8)
        for a, b in EDGES:
            cv2.line(img, tuple(kp[KP[a]].astype(int)), tuple(kp[KP[b]].astype(int)), (200,200,200), 6)
        vw.write(img)
    vw.release()

class FakeEstimator:
    """Returns the synthetic athlete (noisy, with dropouts) plus a bystander."""
    def __init__(self): self.i = 0
    def detect(self, frame):
        kp, sc = FRAMES[min(self.i, len(FRAMES) - 1)]; self.i += 1
        kp = kp + rng.normal(0, 2.0, kp.shape); sc = sc.copy()
        if rng.random() < 0.03: sc[:] = 0.1          # missed detection
        box = np.array([*kp.min(0) - 20, *kp.max(0) + 20])
        by = kp * 0.5 + [650, 330]                    # small person in background
        people = [Person(box, kp, sc), Person(np.array([*by.min(0), *by.max(0)]), by, np.full(17, .9))]
        return people[::-1]
