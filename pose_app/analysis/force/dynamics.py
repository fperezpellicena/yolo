"""Newton-Euler inverse dynamics of the planar chain, from the cord to the floor.

For joint j, the net muscle moment on its hand side (the segments between the
joint and the cord) balances that part's motion, its weight and the cord:

    M_j = sum_k [ I_k alpha_k + (c_k - r_j) x m_k (a_k - g) ]  -  (h - r_j) x F_cord

with c_k, a_k the segment centres of mass and their accelerations, h the grip
and x the 2-D cross product (a_x b_y - a_y b_x). Moments are counter-clockwise
positive in the x-forward, y-up frame, for both sides together (halve them per
side). Over a stroke the joints' work must equal the change in the body's
energy plus the work done on the cord: the energy check.
"""

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .kinematics import Kinematics
from .model import G, HAND_SIDE, JOINT_POINT, JOINT_SEGMENTS, BodyModel

GRAVITY = np.array([0.0, -G])

# Anatomical names of a positive / negative moment (counter-clockwise on the hand side)
MOMENT_NAMES = {"elbow": ("flexion", "extension"), "shoulder": ("flexion", "extension"),
                "hip": ("extension", "flexion"), "knee": ("flexion", "extension"),
                "ankle": ("plantarflexion", "dorsiflexion")}


def cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


@dataclass
class Dynamics:
    moment: Dict[str, np.ndarray]    # joint -> (N,) N m, both sides together
    power: Dict[str, np.ndarray]     # joint -> (N,) W, both sides together
    energy: np.ndarray               # (N,) J: kinetic + potential energy of the moving segments
    floor: np.ndarray                # (N, 2) floor reaction, both feet (not reported yet)
    cop_x: np.ndarray                # (N,) centre of pressure, m from the ankle (not reported yet)

    @property
    def joint_power(self) -> np.ndarray:
        return sum(self.power.values())


def inverse_dynamics(kin: Kinematics, model: BodyModel, cord: np.ndarray) -> Dynamics:
    """`cord`: (N, 2) force of the cords on the hands (both together), newtons."""
    grip = kin.points["grip"]
    load = {n: s.mass * (kin.com_acc[n] - GRAVITY) for n, s in model.segments.items()}
    spin = {n: s.inertia * kin.alpha[n] for n, s in model.segments.items()}
    moment, power = {}, {}
    for joint, segs in HAND_SIDE.items():
        r = kin.points[JOINT_POINT[joint]]
        m = np.zeros(len(kin.t)) if joint == "neck" else -cross(grip - r, cord)
        for n in segs:
            m = m + spin[n] + cross(kin.com[n] - r, load[n])
        moment[joint] = m
        hand, floor = JOINT_SEGMENTS[joint]
        power[joint] = m * (kin.omega[hand] - (kin.omega[floor] if floor else 0.0))
    energy = sum(s.mass * (0.5 * np.sum(kin.com_vel[n] ** 2, axis=1) + G * kin.com[n][:, 1]) +
                 0.5 * s.inertia * kin.omega[n] ** 2 for n, s in model.segments.items())
    feet = -model.feet_mass * GRAVITY
    floor = sum(load.values()) + feet - cord
    turn = sum(spin[n] + cross(kin.com[n], load[n]) for n in model.segments)
    turn = turn + cross(model.feet_com, feet) - cross(grip, cord)
    with np.errstate(invalid="ignore", divide="ignore"):
        cop = turn / floor[:, 1]
    return Dynamics(moment, power, energy, floor, cop)


def static_capacity(com_x: float, grip: np.ndarray, u: np.ndarray,
                    model: BodyModel) -> Tuple[float, str]:
    """Most cord tension this posture holds without accelerating, and the foot edge that
    limits it ("heel" or "toe"; "lift" when the feet unload first).

    Quasi-static, feet flat: the centre of pressure moves with the tension T as
    x(T) = (W x_com - T h x u) / (W - T u_y) and must stay between heel and toe.
    """
    W = model.weight
    if not (np.isfinite(com_x) and np.isfinite(grip).all() and np.isfinite(u).all()):
        return math.nan, ""
    hxu = float(cross(grip, u))
    t_lift = W / u[1] if u[1] > 1e-6 else math.inf
    heel, toe = -model.heel, model.toe

    def reaches(x_lim: float) -> float:
        den = x_lim * u[1] - hxu
        return W * (x_lim - com_x) / den if abs(den) > 1e-9 else math.inf

    rising = com_x * u[1] - hxu > 0                 # the centre of pressure moves to the toe
    if (rising and com_x > toe) or (not rising and com_x < heel):
        return math.nan, ""                         # already past the edge it moves towards
    if rising:
        lo = reaches(heel) if com_x < heel else 0.0
        hi, edge = reaches(toe), "toe"
    else:
        lo = reaches(toe) if com_x > toe else 0.0
        hi, edge = reaches(heel), "heel"
    if not (0 < hi < t_lift):
        hi, edge = t_lift, "lift"
    if not (0 <= lo <= hi) or not math.isfinite(hi):
        return math.nan, ""
    return float(hi), edge


def moment_arm(joint: np.ndarray, grip: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Distance from a joint to the cord's line of action (m)."""
    return np.abs(cross(u, joint - grip))
