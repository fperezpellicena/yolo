"""Physically consistent synthetic SkiErg session for the force-analysis tests.

An 80 kg, 1.80 m athlete moves through smooth, hinge-led strokes (the phase
postures of the requirement analysis) whose rate fades and surges; the cords
pull with a drive-shaped tension. From that one motion come:
  * the keypoints a side-on camera at 300 px/m would see (FakeEstimator), with
    pixel noise, a far side and a few missed detections;
  * the PM5's Bluetooth log: status samples, stroke data twice per stroke,
    force curves split into packets, one repeated notification and one lost
    curve packet, on a clock that started LOG_OFFSET s before the video;
  * the exact values (`Session.truth`) the analysis should recover.
"""
import json, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, cv2

from pose_app.person import Person
from pose_app.skeleton import KP
from pose_app.pm5.protocol import LBF_TO_N, curve_packets
from pose_app.analysis.force.dynamics import inverse_dynamics, static_capacity, moment_arm
from pose_app.analysis.force.kinematics import kinematics
from pose_app.analysis.force.model import BodyModel, G
from pose_app.analysis.force.setup import AthleteProfile, Calibration, ForceSetup

FPS, W, H = 30, 960, 720
SCALE = 300.0                       # px per metre
ORIGIN = (330.0, 700.0)             # px: the floor under the ankle (athlete faces image-right)
MASS, HEIGHT = 80.0, 1.80
CORD_EXIT = np.array([0.75, 2.15])  # m from the floor under the ankle
LOG_OFFSET = 7.43                   # PM5 log seconds at video second 0
EPOCH = 1_791_460_800.0
LEAD_S, TAIL_S = 1.2, 0.8
LEN = dict(ankle_h=0.039, shank=0.246, thigh=0.245, trunk=0.288, neck=0.092,
           upper_arm=0.186, forearm=0.146, grip=0.050)
ATHLETE = AthleteProfile(MASS, HEIGHT, "male")

# Phase postures: shank, thigh, trunk and head angles from vertical (deg), grip position (m)
PHASES = [  # (phase, th_s, th_t, th_k, th_h, hand)
    (0.000, 9, 8, 20, 10, (0.62, 1.90)),      # catch
    (0.087, 13, 20, 32, 18, (0.65, 1.78)),    # early drive
    (0.173, 17, 33, 45, 28, (0.62, 1.56)),    # peak force
    (0.267, 17, 32, 50, 32, (0.70, 1.09)),    # late drive
    (0.367, 17, 30, 52, 34, (0.20, 0.56)),    # finish
    (0.667, 11, 16, 28, 15, (0.76, 1.74)),    # recovery
]
DRIVE = 0.367                       # drive: phase 0 -> DRIVE
HARMONICS = 5


def _lengths():
    return {k: v * HEIGHT for k, v in LEN.items()}


def _chain(th_s, th_t, th_k, th_h):
    L = _lengths()
    r = np.radians
    th_s, th_t, th_k, th_h = (np.asarray(a, float) for a in (th_s, th_t, th_k, th_h))
    ankle = np.stack([np.zeros_like(th_s), np.full_like(th_s, L["ankle_h"])], -1)
    knee = ankle + L["shank"] * np.stack([np.sin(r(th_s)), np.cos(r(th_s))], -1)
    hip = knee + L["thigh"] * np.stack([-np.sin(r(th_t)), np.cos(r(th_t))], -1)
    sh = hip + L["trunk"] * np.stack([np.sin(r(th_k)), np.cos(r(th_k))], -1)
    ear = sh + L["neck"] * np.stack([np.sin(r(th_h)), np.cos(r(th_h))], -1)
    return dict(ankle=ankle, knee=knee, hip=hip, shoulder=sh, ear=ear)


def _tracks():
    """Fourier coefficients of each body track over one cycle (phase 0..1)."""
    keys, prev = [], None
    for ph, s, t, k, h, hand in PHASES:
        sh = _chain(s, t, k, h)["shoulder"]
        v = np.array(hand) - sh
        phi = math.degrees(math.atan2(v[1], v[0]))
        if prev is not None:
            phi = prev + ((phi - prev + 180) % 360 - 180)
        prev = phi
        keys.append((ph, [s, t, k, h, phi, float(np.hypot(*v))]))
    keys.append((1.0, list(keys[0][1])))
    n = 2048
    grid = np.arange(n) / n
    coefs = []
    for c in range(6):
        y = np.empty(n)
        for i, g in enumerate(grid):
            for (p0, v0), (p1, v1) in zip(keys, keys[1:]):
                if p0 <= g <= p1:
                    u = (1 - math.cos(math.pi * (g - p0) / (p1 - p0))) / 2
                    y[i] = v0[c] + (v1[c] - v0[c]) * u
                    break
        f = np.fft.rfft(y) / n
        f[HARMONICS + 1:] = 0
        coefs.append(f)
    return coefs


_COEFS = _tracks()


def _eval(f, phase, order=0):
    k = np.arange(len(f))
    w = (2j * np.pi * k) ** order
    z = np.exp(2j * np.pi * np.outer(np.atleast_1d(phase), k))
    v = (z * (w * f)).real @ np.where(k == 0, 1, 2)
    return v


def pose(phase):
    """Exact joint positions (m) at the given phases."""
    s, t, k, h, phi, d = (_eval(f, phase) for f in _COEFS)
    pts = _chain(s, t, k, h)
    L = _lengths()
    a, b = L["upper_arm"], L["forearm"] + L["grip"]
    sh = pts["shoulder"]
    d = np.minimum(d, 0.995 * (a + b))
    hand = sh + d[:, None] * np.stack([np.cos(np.radians(phi)), np.sin(np.radians(phi))], -1)
    v = hand - sh
    ang = np.arctan2(v[:, 1], v[:, 0])
    alpha = np.arccos(np.clip((a * a + d * d - b * b) / (2 * a * d), -1, 1))
    elbow = sh + a * np.stack([np.cos(ang - alpha), np.sin(ang - alpha)], -1)
    fore = (hand - elbow) / np.linalg.norm(hand - elbow, axis=1, keepdims=True)
    pts.update(elbow=elbow, wrist=elbow + L["forearm"] * fore, grip=hand)
    pts["head_angle"] = np.radians(h)
    return pts


def tension_shape(u, a=1.6, b=2.4):
    tm = a / (a + b)
    u = np.clip(u, 0, 1)
    return (u / tm) ** a * ((1 - u) / (1 - tm)) ** b


class Session:
    def __init__(self, n_strokes=24, travel=False, seed=0, cord_exit=CORD_EXIT,
                 noise_px=1.2, fs=240.0, pattern=(13.0, 4.7, 1.0)):
        """`pattern`: periods (s) and phase of the rate's surges, to make another piece."""
        self.n, self.travel, self.cord_exit = n_strokes, travel, np.asarray(cord_exit, float)
        self.rng = np.random.default_rng(seed)
        self.noise_px = noise_px
        # phase(t): still at the catch, a smooth start, then a fading, surging rate
        dt = 1.0 / fs
        t, ph, phase = [0.0], [0.0], 0.0
        while phase < n_strokes:
            tt = t[-1] + dt
            run = max(0.0, tt - LEAD_S)
            ramp = 0.5 - 0.5 * math.cos(math.pi * min(run / 0.8, 1.0))
            slow, fast, shift = pattern
            period = 1.5 * (1 + 0.006 * run) * (1 + 0.06 * math.sin(2 * math.pi * run / slow)
                                               + 0.04 * math.sin(2 * math.pi * run / fast + shift))
            phase = min(phase + ramp * dt / period, float(n_strokes))
            t.append(tt)
            ph.append(phase)
        t_stop = t[-1]
        while t[-1] < t_stop + TAIL_S:
            t.append(t[-1] + dt)
            ph.append(float(n_strokes))
        self.t_fine, self.phase_fine = np.array(t), np.array(ph)
        self.duration = self.t_fine[-1]
        self.peak_n = np.array([480.0 * (1 - 0.006 * k) * (1 + 0.05 * math.sin(1.7 * k))
                                for k in range(n_strokes)])
        self._fine()

    # ---------- exact motion and force
    def phase_at(self, t):
        return np.interp(t, self.t_fine, self.phase_fine)

    def points_at(self, t):
        return pose(self.phase_at(t))

    def tension_at(self, t):
        ph = self.phase_at(t)
        k = np.minimum(np.floor(ph).astype(int), self.n - 1)
        frac = ph - k
        T = np.where(frac < DRIVE, self.peak_n[k] * tension_shape(frac / DRIVE), 0.0)
        return np.where(ph >= self.n, 0.0, T)

    def _fine(self):
        t = self.t_fine
        pts = pose(self.phase_fine)
        self.pts_fine = pts
        L = np.linalg.norm(self.cord_exit - pts["grip"], axis=1)
        T = self.tension_at(t)
        self.L_fine, self.T_fine = L, T
        k = np.floor(self.phase_fine).astype(int)
        strokes = []
        for s in range(self.n):
            a = int(np.argmax(self.phase_fine > s))
            b = int(np.argmax(self.phase_fine >= s + DRIVE))
            nxt = int(np.argmax(self.phase_fine >= s + 1)) if s + 1 < self.n else len(t) - 1
            on = slice(a, b + 1)
            work = float(np.sum(0.5 * (T[a + 1:b + 1] + T[a:b]) * np.diff(L[on])))
            strokes.append(dict(a=a, b=b, next=nxt, t_start=t[a], t_end=t[b],
                                drive_time=t[b] - t[a], drive_length=L[b] - L[a],
                                work=work, peak=float(T[on].max()),
                                t_peak=float(t[a + int(np.argmax(T[on]))])))
        self.strokes = strokes

    # ---------- what the analysis should find, from the exact motion at 240 Hz
    def truth(self):
        t, pts = self.t_fine, {k: v for k, v in self.pts_fine.items() if k != "head_angle"}
        lens = _lengths()
        model = BodyModel(ATHLETE, {"forearm_hand": lens["forearm"], "upper_arm": lens["upper_arm"],
                                    "trunk": lens["trunk"], "thigh": lens["thigh"],
                                    "shank": lens["shank"]})
        kin = kinematics(pts, t, model)
        u = (self.cord_exit - pts["grip"])
        u = u / np.linalg.norm(u, axis=1, keepdims=True)
        dyn = inverse_dynamics(kin, model, self.T_fine[:, None] * u)
        out = []
        y = kin.body_com[:, 1]
        for s in self.strokes:
            a, b, nxt = s["a"], s["b"], s["next"]
            k = a + int(np.argmax(self.T_fine[a:b + 1]))
            e = slice(a, nxt)
            joints = float(np.sum(0.5 * (dyn.joint_power[e][1:] + dyn.joint_power[e][:-1]) *
                                  np.diff(t[e])))
            gained = dyn.energy[nxt - 1] - dyn.energy[a]
            cord = float(np.sum(0.5 * (self.T_fine[a + 1:nxt] + self.T_fine[a:nxt - 1]) *
                                np.diff(self.L_fine[a:nxt])))
            zone = np.arange(a, b + 1)
            zone = zone[self.T_fine[zone] >= 0.8 * self.T_fine[k]]     # "at peak force"
            w = self.T_fine[zone] / self.T_fine[zone].sum()
            at = lambda v: float(np.sum(np.asarray(v) * w))
            uz, Tz = u[zone], self.T_fine[zone]
            out.append(dict(
                peak_n=s["peak"], t_peak=t[k], work=s["work"],
                bw_share=100 * MASS * G * (y[a] - y[b]) / s["work"],
                cord_angle=at(np.degrees(np.arctan2(uz[:, 1], uz[:, 0]))),
                friction=at(Tz * np.abs(uz[:, 0]) / (model.weight - Tz * uz[:, 1])),
                capacity=at([static_capacity(kin.body_com[i, 0], pts["grip"][i], u[i], model)[0]
                             for i in zone]),
                shoulder_ext=at(-dyn.moment["shoulder"][zone] / 2),
                elbow_flex=at(dyn.moment["elbow"][zone] / 2),
                shoulder_arm=100 * at(moment_arm(pts["shoulder"][zone], pts["grip"][zone], uz)),
                energy_err=100 * (joints - gained - cord) / cord))
        return out

    # ---------- the camera
    def frame_times(self):
        return np.arange(int(self.duration * FPS)) / FPS

    def keypoints(self):
        """(frames, 17, 2) pixel keypoints and (frames, 17) scores, noise and misses included."""
        t = self.frame_times()
        pts = self.points_at(t)
        px = lambda m: np.stack([ORIGIN[0] + m[..., 0] * SCALE, ORIGIN[1] - m[..., 1] * SCALE], -1)
        n = len(t)
        kp, sc = np.zeros((n, 17, 2)), np.full((n, 17), 0.9)
        for j in ("shoulder", "elbow", "wrist", "hip", "knee", "ankle", "ear"):
            p = px(pts[j])
            kp[:, KP[f"left_{j}"]] = p
            kp[:, KP[f"right_{j}"]] = p + [4, -2]
            sc[:, KP[f"right_{j}"]] = 0.6
        h = pts["head_angle"]
        fwd = np.stack([np.cos(h), -np.sin(h)], -1)          # face direction, tilting with the head
        ear = pts["ear"]
        kp[:, KP["nose"]] = px(ear + 0.10 * fwd - [0, 0.03])
        kp[:, KP["left_eye"]] = px(ear + 0.08 * fwd + [0, 0.02])
        kp[:, KP["right_eye"]] = kp[:, KP["left_eye"]] + [3, 0]
        kp += self.rng.normal(0, self.noise_px, kp.shape)
        miss = self.rng.random(n) < 0.01
        sc[miss] = 0.1
        return kp, sc

    def estimator(self):
        kp, sc = self.keypoints()
        return FakeEstimator(kp, sc)

    def write_video(self, path):
        kp, _ = self.keypoints()
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
        for f in range(len(kp)):
            img = np.full((H, W, 3), 50, np.uint8)
            for a, b in (("shoulder", "elbow"), ("elbow", "wrist"), ("shoulder", "hip"),
                         ("hip", "knee"), ("knee", "ankle")):
                cv2.line(img, tuple(kp[f, KP[f"left_{a}"]].astype(int)),
                         tuple(kp[f, KP[f"left_{b}"]].astype(int)), (200, 200, 200), 5)
            vw.write(img)
        vw.release()

    def setup(self, scale=True, cord_exit=None):
        ex = self.cord_exit if cord_exit is None else np.asarray(cord_exit, float)
        exit_px = (ORIGIN[0] + ex[0] * SCALE, ORIGIN[1] - ex[1] * SCALE)
        stick = ((250.0, ORIGIN[1] - 0.2 * SCALE), (250.0, ORIGIN[1] - 1.2 * SCALE))
        return ForceSetup(ATHLETE, Calibration(exit_px, stick if scale else None,
                                               1.0 if scale else None))

    # ---------- the PM5 log
    def pm5_rows(self, duplicate=5, lose_packet=9):
        """[(log seconds, short id, payload)], unsorted; strokes are 1-based on the PM5."""
        rows, rng = [], self.rng
        log = lambda video_t: video_t + LOG_OFFSET
        zero = log(self.strokes[0]["t_start"])          # the PM5 clock starts with the piece
        el = lambda lt: max(0.0, round(lt - zero + 0.02, 2))
        u24 = lambda v: int(round(v)).to_bytes(3, "little")
        u16 = lambda v: int(round(v)).to_bytes(2, "little")
        dist = 0.0
        recovery_prev = 0.0
        for k, s in enumerate(self.strokes):
            period = (self.strokes[k + 1]["t_start"] if k + 1 < self.n else s["t_end"] + 0.9) - s["t_start"]
            stroke_dist = 10.0 * (s["work"] / 300.0) ** (1 / 3)
            dist += stroke_dist
            recovery = period - s["drive_time"]
            arrive = log(s["t_end"]) + 0.035 + rng.normal(0, 0.008)

            def stroke_data(lt, rec):
                return (u24(el(lt) * 100) + u24(dist * 10) + bytes([int(round(s["drive_length"] * 100)),
                        int(round(s["drive_time"] * 100))]) + u16(rec * 100) + u16(stroke_dist * 100) +
                        u16(s["peak"] / LBF_TO_N * 10) +
                        u16(s["work"] / s["drive_length"] / LBF_TO_N * 10) + u16(s["work"] * 10) + u16(k + 1))
            first = stroke_data(log(s["t_end"]), recovery_prev)
            rows.append((arrive, 0x0035, first))
            if k + 1 == duplicate:
                rows.append((arrive + 0.004, 0x0035, first))
            rows.append((arrive + 0.003, 0x0036, u24(el(log(s["t_end"])) * 100) +
                         u16(s["work"] / period) + u16(900) + u16(k + 1)))
            if k + 1 < self.n:
                nxt = log(self.strokes[k + 1]["t_start"]) + 0.03 + rng.normal(0, 0.008)
                rows.append((nxt, 0x0035, stroke_data(log(self.strokes[k + 1]["t_start"]), recovery)))
            recovery_prev = recovery
            # force against time: steps of 13.5 ms, each reading sent twice, zeros first
            tt = np.arange(s["t_start"], s["t_end"], 0.027)
            pts_lbf = [0, 0] + [int(round(v / LBF_TO_N)) for v in self.tension_at(tt) for _ in (0, 1)]
            packets = curve_packets(pts_lbf)
            for i, p in enumerate(packets):
                if k + 1 == lose_packet and i == 1:
                    continue
                rows.append((arrive + 0.010 + 0.002 * i, 0x003D, p))
            if self.travel:     # force against handle travel, one point per 2.96 cm
                Lw = self.L_fine[s["a"]:s["b"] + 1]
                tw = self.t_fine[s["a"]:s["b"] + 1]
                steps = np.arange(Lw[0], Lw[-1], 0.0296)
                when = np.interp(steps, np.maximum.accumulate(Lw), tw)
                vals = [int(round(v / LBF_TO_N)) for v in self.tension_at(when)]
                for i, p in enumerate(curve_packets(vals)):
                    rows.append((arrive + 0.03 + 0.002 * i, 0x0043, p))
        # status twice a second from the moment the log starts
        end = log(self.duration) + 1.0
        for lt in np.arange(0.2, end, 0.5):
            vt = lt - LOG_OFFSET
            done = [s for s in self.strokes if s["t_end"] <= vt]
            d = sum(10.0 * (s["work"] / 300.0) ** (1 / 3) for s in done)
            rows.append((lt, 0x0031, u24(el(lt) * 100) + u24(d * 10) +
                         bytes([1, 0, 1, 1, 1]) + u24(0) + u24(0) + bytes([0, 110])))
            spm = 60.0 / 1.5 if done else 0
            rows.append((lt + 0.05, 0x0032, u24(el(lt) * 100) + u16(4200 if done else 0) +
                         bytes([int(spm), 150]) + u16(11900) + u16(11900) + u16(0) + u24(0)))
        return rows

    def write_pm5_log(self, path, **kw):
        rows = sorted(self.pm5_rows(**kw), key=lambda r: r[0])
        with open(path, "w") as fh:
            fh.write(json.dumps({"t": EPOCH, "device": {"name": "PM5 431234567 Ski",
                                                        "serial": "431234567",
                                                        "firmware_rev": "synthetic"}}) + "\n")
            for t, short, payload in rows:
                fh.write(json.dumps({"t": round(EPOCH + t, 3), "uuid": f"{short:04x}",
                                     "hex": payload.hex()}) + "\n")


class FakeEstimator:
    """Returns the synthetic athlete frame by frame."""
    def __init__(self, kp, sc):
        self.kp, self.sc, self.i = kp, sc, 0

    def detect(self, frame):
        i = min(self.i, len(self.kp) - 1)
        self.i += 1
        kp = self.kp[i]
        box = np.array([*np.nanmin(kp, 0) - 20, *np.nanmax(kp, 0) + 20])
        return [Person(box, kp.copy(), self.sc[i].copy())]
