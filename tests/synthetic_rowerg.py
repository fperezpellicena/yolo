"""Synthetic RowErg athlete (side view, facing image-right, flywheel to the right).

The ankle sits still on the footplate; the seat slides on a rail 50 px above it; the
handle runs level at 400 px. Each stroke is built from the seat's position, the trunk's
lean and the elbow's angle over time, the knee and elbow placed by two-link geometry.
Torso 200 px (about 0.5 m). Kinds:
  good    legs, then trunk once the knees pass 90, then arms; hands past the knees before
          the knees bend, body swung over by halfway up the slide
  arms    elbows bend at the start of the drive
  shoot   the seat runs back while the shoulders stay: the trunk folds forward
  body    the trunk opens from the catch, with the legs
  knees   the knees rise straight after the finish, before the hands pass them
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, cv2
from pose_app.person import Person
from pose_app.skeleton import KP
from pose_app.analysis.render import EDGES

FPS, W, H = 30, 960, 720
ANKLE = np.array([640.0, 540.0])
SEAT_Y = ANKLE[1] - 50              # hip height on the seat
HANDLE_Y = 400.0
SHANK, THIGH, TRUNK, UPPER, FORE = 150.0, 160.0, 200.0, 110.0, 100.0
CATCH_X, FINISH_X = -125.0, -303.0  # hip ahead of the ankle (px): shins vertical / legs down
DRIVE_S = 0.95


def interp(u, keys):  # keys: (phase, value); smooth piecewise cosine
    for (p0, v0), (p1, v1) in zip(keys, keys[1:]):
        if p0 <= u <= p1:
            f = (u - p0) / (p1 - p0) if p1 > p0 else 1.0
            return v0 + (v1 - v0) * (1 - np.cos(np.pi * f)) / 2
    return keys[-1][1] if u > keys[-1][0] else keys[0][1]


def two_link(a, b, la, lb, up):
    """Joint between a and b (link lengths la from a, lb from b); `up` picks the solution
    above the line a-b (smaller image y), else below."""
    d = b - a
    D = min(np.linalg.norm(d), la + lb - 1e-6)
    ca = np.clip((la ** 2 + D ** 2 - lb ** 2) / (2 * la * D), -1, 1)
    ang = np.arccos(ca)
    base = np.arctan2(d[1], d[0])
    cands = [a + la * np.array([np.cos(base + s * ang), np.sin(base + s * ang)]) for s in (1, -1)]
    return min(cands, key=lambda p: p[1]) if up else max(cands, key=lambda p: p[1])


DRIVE = {   # v: 0 catch -> 1 finish
    "seat": {"good": [(0, CATCH_X), (0.65, FINISH_X)], "shoot": [(0, CATCH_X), (0.45, FINISH_X)]},
    "trunk": {"good": [(0, 25), (0.30, 25), (0.85, -25)],
              "shoot": [(0, 25), (0.30, 42), (0.50, 30), (0.90, -25)],
              "body": [(0, 25), (0.60, -25)]},
    "elbow": {"good": [(0, 172), (0.55, 172), (1.0, 65)],
              "arms": [(0, 172), (0.08, 172), (0.45, 95), (1.0, 65)]},
}
RECOVERY = {   # w: 0 finish -> 1 next catch
    "seat": {"good": [(0, FINISH_X), (0.32, FINISH_X), (1.0, CATCH_X)],
             "knees": [(0, FINISH_X), (0.04, FINISH_X), (1.0, CATCH_X)]},
    "trunk": {"good": [(0, -25), (0.10, -25), (0.45, 25)]},
    "elbow": {"good": [(0, 65), (0.20, 172)], "knees": [(0, 65), (0.35, 172)]},
}


def body_at(phase, kind, drive_frac):
    table, u = (DRIVE, phase / drive_frac) if phase < drive_frac else \
        (RECOVERY, (phase - drive_frac) / (1 - drive_frac))
    seat, trunk, elbow = (interp(u, table[k].get(kind, table[k]["good"]))
                          for k in ("seat", "trunk", "elbow"))
    return seat, trunk, elbow


def pose(seat, trunk, elbow):
    d = np.radians
    hip = np.array([ANKLE[0] + seat, SEAT_Y])
    knee = two_link(ANKLE, hip, SHANK, THIGH, up=True)
    sh = hip + TRUNK * np.array([np.sin(d(trunk)), -np.cos(d(trunk))])
    reach = np.sqrt(UPPER ** 2 + FORE ** 2 - 2 * UPPER * FORE * np.cos(d(elbow)))
    dy = HANDLE_Y - sh[1]
    dx = np.sqrt(max(reach ** 2 - dy ** 2, 1.0))
    wrist = np.array([sh[0] + dx, sh[1] + dy])
    elb = two_link(sh, wrist, UPPER, FORE, up=False)
    head = sh + 55 * np.array([np.sin(d(trunk)), -np.cos(d(trunk))])
    kp = np.zeros((17, 2)); sc = np.full(17, 0.9)
    near = dict(shoulder=sh, elbow=elb, wrist=wrist, hip=hip, knee=knee, ankle=ANKLE)
    for j, p in near.items():
        kp[KP[f"left_{j}"]] = p; kp[KP[f"right_{j}"]] = p + [6, -2]; sc[KP[f"right_{j}"]] = 0.6
    kp[KP["left_ear"]] = head; kp[KP["right_ear"]] = head + [4, 0]; sc[KP["right_ear"]] = 0.5
    kp[KP["nose"]] = head + [22, 6]; kp[KP["left_eye"]] = head + [16, -2]
    kp[KP["right_eye"]] = head + [18, -2]
    return kp, sc


LEAD_S = 1.0                             # athlete still at the catch before stroke 1


def period_of(n):
    """Seconds for stroke n (0-based): the rate fades from 25 to 20 spm."""
    return 2.4 + 0.03 * n


def build():
    kinds = ["good"] * 6 + ["arms"] * 3 + ["shoot"] * 3 + ["body"] * 3 + ["knees"] * 3
    catch = pose(*body_at(0.0, "good", 0.4))
    frames = [catch] * int(LEAD_S * FPS)
    starts = []
    for n, kind in enumerate(kinds):
        period = period_of(n)
        nf = int(period * FPS)
        starts.append(len(frames) / FPS)
        for i in range(nf):
            frames.append(pose(*body_at(i / nf, kind, DRIVE_S / period)))
    frames += [catch] * 10
    return frames, kinds, starts


FRAMES, KINDS, STROKE_STARTS = build()


def write_video(path):
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for kp, sc in FRAMES:
        img = np.full((H, W, 3), 60, np.uint8)
        for a, b in EDGES:
            if max(KP[a], KP[b]) >= len(kp):        # the synthetic athlete has no feet
                continue
            cv2.line(img, tuple(kp[KP[a]].astype(int)), tuple(kp[KP[b]].astype(int)),
                     (200, 200, 200), 6)
        vw.write(img)
    vw.release()


class FakeEstimator:
    """Returns the synthetic athlete (noisy, with dropouts) plus a bystander."""
    def __init__(self, seed=0): self.i = 0; self.rng = np.random.default_rng(seed)
    def detect(self, frame):
        kp, sc = FRAMES[min(self.i, len(FRAMES) - 1)]; self.i += 1
        kp = kp + self.rng.normal(0, 2.0, kp.shape); sc = sc.copy()
        if self.rng.random() < 0.03: sc[:] = 0.1     # missed detection
        box = np.array([*kp.min(0) - 20, *kp.max(0) + 20])
        by = kp * 0.4 + [40, 330]                     # small person in the background
        people = [Person(box, kp, sc), Person(np.array([*by.min(0), *by.max(0)]), by,
                                               np.full(17, .9))]
        return people[::-1]
