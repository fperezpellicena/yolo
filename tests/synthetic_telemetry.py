"""Synthetic Track My Indoor Workout CSV matching the synthetic SkiErg clip.

Reproduces what real files showed: ~0.48 s logging grid, values updated once
per stroke and smoothed over several strokes, integer watts / spm, distance in
3-4 m steps, held values after the athlete stops, and the app's header.
"""
import numpy as np

EPOCH_MS = 1_790_431_415_739
DT = 0.4815


def make_tmiw_csv(path, stroke_starts, periods, kinds, log_start,
                  stale_tail_s=6.0, hr=None, seed=1):
    """Write a CSV whose first sample is at video time `log_start`.

    So the true offset (machine s at video 0) is -log_start. Times below are
    video times. Power drops 20% on 'squat' strokes to create a late fade.
    """
    rng = np.random.default_rng(seed)
    ends = np.array(stroke_starts) + np.array(periods)
    t_end = ends[-1] + stale_tail_s
    t = np.arange(log_start, t_end, DT)
    t = t + np.concatenate([[0.0], rng.normal(0, 0.002, len(t) - 1)])

    raw_p = np.array([190.0 * (0.8 if k == "squat" else 1.0) for k in kinds])
    raw_r = 60.0 / np.array(periods)
    sm_p, sm_r = [], []
    ep, er = raw_p[0], raw_r[0]
    for p, r in zip(raw_p, raw_r):                 # monitor-style smoothing
        ep, er = 0.5 * ep + 0.5 * p, 0.6 * er + 0.4 * r
        sm_p.append(ep)
        sm_r.append(er)

    rows, dist = [], 0.0
    for ti in t:
        k = np.searchsorted(ends, ti) - 1          # last completed stroke
        moving = stroke_starts[0] <= ti <= ends[-1]
        if k < 0:
            power, spm = 60.0, 40.0                # first packet before stroke 1 completes
        else:
            power, spm = sm_p[k], sm_r[k]
        speed = (power / 2.8) ** (1 / 3) * 3.6     # km/h
        if moving:
            dist += speed / 3.6 * DT
        heart = hr if hr is not None else 100 + ti - log_start
        rows.append((int(round(power)), int(round(spm)), int(heart), float(np.floor(dist)),
                     int(EPOCH_MS + (ti - log_start) * 1000), int(ti - log_start),
                     round(speed, 2), 0))   # held when idle

    lines = ["TMIW,5,", "RIDE SUMMARY,", f"Total Time,{int(t[-1] - t[0])},Seconds,",
             f"Total Distance,{rows[-1][3]:.1f},M,", "Device Name,SYNTH-SKI,",
             f"Start Time,{rows[0][4]},", "Sport,Rowing,", "Power Factor,1.0,",
             "Time Zone,Europe/Madrid,", "", "RIDE DATA",
             "Power,RPM,HR,DISTANCE,TIME_STAMP,ELAPSED,SPEED,CAL,"]
    lines += [f"{p},{r},{h},{d:.2f},{ts},{e},{s},{c}," for p, r, h, d, ts, e, s, c in rows]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
