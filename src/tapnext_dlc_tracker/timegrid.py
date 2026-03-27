from __future__ import annotations


def make_windows(start_s: float, end_s: float, window_s: float) -> list[tuple[float, float]]:
    if window_s <= 0:
        raise ValueError("window_s must be > 0")
    if end_s <= start_s:
        return []

    windows: list[tuple[float, float]] = []
    cur = start_s
    while cur < end_s:
        nxt = min(end_s, cur + window_s)
        windows.append((cur, nxt))
        cur = nxt
    return windows


def make_supervision_times(start_s: float, end_s: float, stride_s: float) -> list[float]:
    if stride_s <= 0:
        raise ValueError("stride_s must be > 0")
    if end_s <= start_s:
        return []

    times: list[float] = []
    t = start_s
    while t <= end_s + 1e-9:
        times.append(round(t, 6))
        t += stride_s
    return times

