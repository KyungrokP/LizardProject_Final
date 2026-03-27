from __future__ import annotations

import math

from .types import QueryFrame, TrackPoint


def apply_supervision(
    points: list[TrackPoint],
    supervision: QueryFrame,
    tapnext_conf_floor: float,
    dlc_conf_floor: float,
    tolerance_s: float,
    dlc_force_replace: bool = False,
    dlc_max_jump_pixels: float | None = None,
) -> list[TrackPoint]:
    if not points:
        return points

    updated = points.copy()
    for i, pt in enumerate(points):
        if abs(pt.timestamp_s - supervision.timestamp_s) > tolerance_s:
            continue
        sup_xy = supervision.points.get(pt.keypoint)
        sup_conf = supervision.likelihoods.get(pt.keypoint, 0.0)
        if sup_xy is None or sup_conf < dlc_conf_floor:
            continue

        if not dlc_force_replace and pt.likelihood >= tapnext_conf_floor:
            continue

        if dlc_max_jump_pixels is not None:
            jump = math.hypot(float(sup_xy[0]) - pt.x, float(sup_xy[1]) - pt.y)
            if jump > dlc_max_jump_pixels:
                continue

        updated[i] = TrackPoint(
            keypoint=pt.keypoint,
            timestamp_s=pt.timestamp_s,
            x=float(sup_xy[0]),
            y=float(sup_xy[1]),
            likelihood=float(sup_conf),
            source="deeplabcut",
        )
    return updated
