"""Stage 5 — navigation slice.

A minimal waypoint navigator that consumes the *estimated* pose (x, y, yaw) and emits a
`VelocityCommand(vx, vy, vyaw)` — the same body-frame command a real humanoid control stack accepts.
It closes the loop **estimator → planner → command**: `scripts/run_nav.py` drives the
walking sim from the robot's own onboard estimate, so estimation error shows up directly as navigation
error (which is exactly why the yaw-fixing VIO matters here).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VelocityCommand:
    vx: float          # forward velocity, body frame (m/s)
    vy: float          # lateral velocity, body frame (m/s)
    vyaw: float        # yaw rate (rad/s)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class WaypointNavigator:
    """Go-to-waypoint list controller. Cruises forward, steers yaw proportionally toward the active
    waypoint, eases off forward speed while turning, and advances when within `reach` of the goal."""

    def __init__(self, waypoints, cruise=0.8, reach=0.35, kp_yaw=1.8, max_vyaw=1.2):
        self.wps = [np.asarray(w, float) for w in waypoints]
        self.i = 0
        self.cruise, self.reach = cruise, reach
        self.kp_yaw, self.max_vyaw = kp_yaw, max_vyaw
        self.desired_heading = 0.0

    @property
    def done(self):
        return self.i >= len(self.wps)

    def command(self, x, y, yaw):
        """VelocityCommand from the estimated pose. Advances the waypoint index as goals are reached."""
        if self.done:
            return VelocityCommand(0.0, 0.0, 0.0)
        gx, gy = self.wps[self.i]
        if np.hypot(gx - x, gy - y) < self.reach:            # reached — advance
            self.i += 1
            if self.done:
                return VelocityCommand(0.0, 0.0, 0.0)
            gx, gy = self.wps[self.i]
        self.desired_heading = float(np.arctan2(gy - y, gx - x))
        eyaw = wrap(self.desired_heading - yaw)
        vyaw = float(np.clip(self.kp_yaw * eyaw, -self.max_vyaw, self.max_vyaw))
        vx = self.cruise * max(0.15, float(np.cos(eyaw)))    # slow down while turning toward the goal
        return VelocityCommand(vx, 0.0, vyaw)
