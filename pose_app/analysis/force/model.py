"""Planar body model: rigid segments built from the keypoints, sized to the athlete.

Segment masses, centres of mass and radii of gyration are de Leva's (1996)
adjustment of Zatsiorsky-Seluyanov's parameters, by sex, as tabulated by
HAS-Motion (Visual3D wiki, "Adjusted Zatsiorsky-Seluyanov's segment inertia
parameters"). Proportions that the keypoints cannot measure (ankle height,
foot, hand-to-grip) are Drillis and Contini's fractions of stature (Winter,
Biomechanics and Motor Control of Human Movement, 2009).

Mapping onto COCO keypoints (approximations, documented in the README):
  * trunk: hip -> shoulder keypoints, the shoulder taken as the cervicale;
  * head and neck: shoulder -> ear, centre of mass at the ear;
  * forearm and hand: one rigid segment; the grip is a hand length beyond the
    wrist along the forearm, the hand's centre of mass at 79% (75% women) of it;
  * feet: flat and static; they add weight but no motion.
Limbs count twice (left and right move together).
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .setup import AthleteProfile

G = 9.81

# name: (mass fraction of the body, centre of mass as a fraction of the length from the
# proximal / cranial end, sagittal radius of gyration as a fraction of the length)
DE_LEVA: Dict[str, Dict[str, Tuple[float, float, float]]] = {
    "male": {"head": (0.0694, 0.5002, 0.303), "trunk": (0.4346, 0.5138, 0.328),
             "upper_arm": (0.0271, 0.5772, 0.285), "forearm": (0.0162, 0.4574, 0.276),
             "hand": (0.0061, 0.7900, 0.628), "thigh": (0.1416, 0.4095, 0.329),
             "shank": (0.0433, 0.4395, 0.251), "foot": (0.0137, 0.4415, 0.257)},
    "female": {"head": (0.0668, 0.4841, 0.271), "trunk": (0.4257, 0.4964, 0.307),
               "upper_arm": (0.0255, 0.5754, 0.278), "forearm": (0.0138, 0.4559, 0.261),
               "hand": (0.0056, 0.7474, 0.631), "thigh": (0.1478, 0.3612, 0.369),
               "shank": (0.0481, 0.4352, 0.267), "foot": (0.0129, 0.4014, 0.299)},
}

# Fractions of stature
ANKLE_H = 0.039          # floor to ankle
HEEL = 0.035             # ankle to heel, horizontally
TOE = 0.117              # ankle to toe tip (foot length 0.152)
GRIP = 0.050             # wrist to the handle's centre in the fist
NECK = 0.092             # shoulder to ear, when the ear is not seen
HEAD = 0.140             # vertex to cervicale: the head's length in the table above
# Stature fractions of the measured segments, for a scale estimate without a stick
SEGMENT_H = {"shank": 0.246, "thigh": 0.245, "trunk": 0.288, "upper_arm": 0.186, "forearm": 0.146}

# Moving segments: proximal point -> distal point (the centre of mass lies on this line,
# and the line's angle is the segment's angle), and how many the body has.
SEGMENTS = {
    "forearm_hand": ("elbow", "wrist", 2),
    "upper_arm": ("shoulder", "elbow", 2),
    "head": ("shoulder", "ear", 1),
    "trunk": ("hip", "shoulder", 1),
    "thigh": ("hip", "knee", 2),
    "shank": ("knee", "ankle", 2),
}
# Joints and the segments on their hand side (between the joint and the cord); the
# neck is listed for the energy balance only.
HAND_SIDE = {
    "elbow": ("forearm_hand",),
    "shoulder": ("forearm_hand", "upper_arm"),
    "neck": ("head",),
    "hip": ("forearm_hand", "upper_arm", "head", "trunk"),
    "knee": ("forearm_hand", "upper_arm", "head", "trunk", "thigh"),
    "ankle": ("forearm_hand", "upper_arm", "head", "trunk", "thigh", "shank"),
}
JOINT_POINT = {"elbow": "elbow", "shoulder": "shoulder", "neck": "shoulder", "hip": "hip",
               "knee": "knee", "ankle": "ankle"}
# (hand-side segment, floor-side segment) whose relative rotation the joint moment works on
JOINT_SEGMENTS = {"elbow": ("forearm_hand", "upper_arm"), "shoulder": ("upper_arm", "trunk"),
                  "neck": ("head", "trunk"), "hip": ("trunk", "thigh"),
                  "knee": ("thigh", "shank"), "ankle": ("shank", None)}
# Joints whose moments one side-on camera measures well enough to report (see the README):
# the lower body waits for the validation study.
REPORTED_JOINTS = ("elbow", "shoulder")


def segment_table(sex: str) -> Dict[str, Tuple[float, float, float]]:
    if sex in DE_LEVA:
        return DE_LEVA[sex]
    m, f = DE_LEVA["male"], DE_LEVA["female"]
    return {k: tuple((a + b) / 2 for a, b in zip(m[k], f[k])) for k in m}


@dataclass(frozen=True)
class Segment:
    mass: float              # kg, all copies together (both arms, both legs)
    com: float               # centre of mass, fraction of proximal -> distal
    inertia: float           # kg m^2 about the centre of mass, all copies together


class BodyModel:
    """Segment masses and inertias for one athlete, from session-median segment lengths."""

    def __init__(self, athlete: AthleteProfile, lengths: Dict[str, float]):
        """`lengths`: metres, measured between the keypoints of SEGMENTS (forearm = elbow-wrist)."""
        self.athlete = athlete
        H, M = athlete.height_m, athlete.mass_kg
        tab = segment_table(athlete.sex)
        self.mass = M
        self.weight = M * G
        self.ankle_h, self.heel, self.toe = ANKLE_H * H, HEEL * H, TOE * H
        self.grip = GRIP * H
        self.neck = NECK * H

        def seg(name: str, length: float, com: float = None, count: int = 1) -> Segment:
            mf, cf, rf = tab[name]
            m = count * mf * M
            return Segment(m, cf if com is None else com, m * (rf * length) ** 2)

        segs: Dict[str, Segment] = {}
        # forearm and hand as one rigid segment along elbow -> wrist
        fa, ha = tab["forearm"], tab["hand"]
        L, g = lengths["forearm_hand"], self.grip
        m_f, m_h = 2 * fa[0] * M, 2 * ha[0] * M
        d_f, d_h = fa[1] * L, L + ha[1] * g              # from the elbow
        d = (m_f * d_f + m_h * d_h) / (m_f + m_h)
        inertia = (m_f * ((fa[2] * L) ** 2 + (d_f - d) ** 2) +
                   m_h * ((ha[2] * g) ** 2 + (d_h - d) ** 2))
        segs["forearm_hand"] = Segment(m_f + m_h, d / L, inertia)
        segs["upper_arm"] = seg("upper_arm", lengths["upper_arm"], count=2)
        head = tab["head"]
        segs["head"] = Segment(head[0] * M, 1.0, head[0] * M * (head[2] * HEAD * H) ** 2)
        segs["trunk"] = seg("trunk", lengths["trunk"], com=1.0 - tab["trunk"][1])   # from the hip
        segs["thigh"] = seg("thigh", lengths["thigh"], count=2)
        segs["shank"] = seg("shank", lengths["shank"], count=2)
        self.segments = segs
        foot = tab["foot"]
        self.feet_mass = 2 * foot[0] * M
        self.feet_com = np.array([-self.heel + foot[1] * (self.heel + self.toe), self.ankle_h / 2])

    @property
    def total_mass(self) -> float:
        return sum(s.mass for s in self.segments.values()) + self.feet_mass
