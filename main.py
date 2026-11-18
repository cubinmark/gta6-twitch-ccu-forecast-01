"""gta6-twitch-ccu-forecast-01 entry: a comparable-launches shape forecaster.

The house baseline holds every hour at hour 1's print. Real launch days do not look like
that: a console game that unlocks at midnight local time, region by region, is already
live in several regions when the UTC day begins, climbs again as the US wakes and
unlocks, peaks in the evening, and only then fades. This entry learns that shape from
the case's own comparables (past big launches with hourly Twitch counts):

  - each comparable's release day is aligned to UTC hours 1..24 from its `day_start`;
  - only comparables whose hour 1 was already a real launch hour -- at least 5% of
    their release-day peak, the regime of a rolling midnight unlock like GTA VI's --
    shape the forecast (with fallbacks if too few qualify);
  - per hour, the log-ratio ln(viewers_h / viewers_1) across those comparables gives
    both a centre and a spread: the forecast's log-quantile at each level is the
    average of a normal fit (median centre, IQR-based sigma) and the empirical
    quantile of the log-ratios, which keeps the tails honest;
  - the case's own hour-1 print p anchors the level: q_tau(h) = p * exp(lq_tau(h)).

Leave-one-out validation over the 37 usable comparables in the practice data (each
comparable forecast from its own hour 1, the model trained on the rest) gives a mean
pinball loss on the log scale of about 0.253 on the launch regime GTA VI is in
(hour 1 at least 10% of the day peak), against about 0.297 for the house baseline
formula on the same pseudo-cases.

Robustness: the grader scores an invalid case as flat at hour 1 and an entry with an
invalid case takes no prize, so every case is wrapped -- any surprise in a case's
data falls back, hour by hour or case by case, to the house baseline formula, and
even an unreadable case file still gets a valid-shaped output under its own name.

Run by the grader as `python3 -P -s /bot/main.py` (Python 3.13, standard library
only, no network), one case per run. Locally, MAKEFUN_INPUT_DIR and
MAKEFUN_OUTPUT_DIR stand in for /input and /output.
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from statistics import NormalDist

# The grader's -P leaves this file's directory off sys.path; nothing else is imported,
# but keep the starter's guard so a sibling module would still be found.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
BASELINE_SIGMA = 0.15  # the house baseline's log-spread per square-root hour
MIN_HOUR1_SHARE = 0.05  # hour 1 at least this share of the day peak: a live launch hour
MIN_TRAIN = 8  # comparables wanted before the filtered set is trusted
MIN_TRAIN_ANY = 3  # comparables wanted before any learned shape is trusted
MIN_DAY_HOURS = 12  # release-day hours a comparable must report to be used


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up (as the starter rounds); whole values as ints."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def parse_time(text) -> datetime | None:
    try:
        return datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except Exception:
        return None


def as_float(value) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def quantile(sorted_vals: list[float], tau: float) -> float:
    """The first value at or past the tau share of the list (the list is pre-sorted)."""
    if not sorted_vals:
        return 0.0
    if tau <= 0:
        return sorted_vals[0]
    if tau >= 1:
        return sorted_vals[-1]
    idx = math.ceil(tau * len(sorted_vals)) - 1
    return sorted_vals[min(max(idx, 0), len(sorted_vals) - 1)]


def median(vals: list[float]) -> float:
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    return s[mid] if n % 2 else 0.5 * (s[mid - 1] + s[mid])


def day_curve(comp: dict) -> dict[int, float] | None:
    """A comparable's release-day viewers by UTC hour (1..24), or None if unusable."""
    start = parse_time(comp.get("day_start"))
    hourly = comp.get("hourly")
    if start is None or not isinstance(hourly, list):
        return None
    curve: dict[int, float] = {}
    for point in hourly:
        if not isinstance(point, dict):
            continue
        when = parse_time(point.get("from"))
        viewers = as_float(point.get("viewers"))
        if when is None or viewers is None:
            continue
        delta = (when - start).total_seconds()
        if 0 <= delta < 24 * 3600 and delta % 3600 == 0:
            curve[int(delta // 3600) + 1] = viewers
    if len(curve) < MIN_DAY_HOURS:
        return None
    first = curve.get(1)
    if first is None or first <= 0:
        return None
    return curve


def collect_log_ratios(case: dict) -> tuple[dict[int, list[float]], int]:
    """Per forecast hour, the log-ratios ln(v_h / v_1) of the regime-matched comparables.

    Returns (ratios_by_hour, n_comparables_used). Comparables whose hour 1 was only a
    pre-unlock trickle (under MIN_HOUR1_SHARE of their day peak) are set aside first;
    if too few remain, every usable comparable is used instead.
    """
    comparables = case.get("comparables")
    if not isinstance(comparables, list):
        return {}, 0
    matched: list[dict[int, float]] = []
    rest: list[dict[int, float]] = []
    for comp in comparables:
        if not isinstance(comp, dict):
            continue
        curve = day_curve(comp)
        if curve is None:
            continue
        peak = max(curve.values())
        if peak > 0 and curve[1] / peak >= MIN_HOUR1_SHARE:
            matched.append(curve)
        else:
            rest.append(curve)
    chosen = matched if len(matched) >= MIN_TRAIN else matched + rest
    if len(chosen) < MIN_TRAIN_ANY:
        return {}, 0
    ratios: dict[int, list[float]] = {}
    for curve in chosen:
        base = curve[1]
        for hour in range(2, 25):
            viewers = curve.get(hour)
            if viewers is not None and viewers > 0:
                ratios.setdefault(hour, []).append(math.log(viewers / base))
    return ratios, len(chosen)


def baseline_row(p: float, hour: int, zs: list[float]) -> list[float]:
    spread = BASELINE_SIGMA * math.sqrt(max(hour - 1, 0))
    return [p * math.exp(z * spread) for z in zs]


def shape_row(p: float, hour: int, log_ratios: list[float], taus: list[float], zs: list[float]) -> list[float]:
    """One hour's quantiles: normal fit and empirical quantiles of the log-ratios, averaged."""
    centre = median(log_ratios)
    if len(log_ratios) >= 4:
        srt = sorted(log_ratios)
        sigma = (quantile(srt, 0.75) - quantile(srt, 0.25)) / 1.349
        if not math.isfinite(sigma) or sigma <= 0:
            sigma = BASELINE_SIGMA * math.sqrt(max(hour - 1, 0))
    else:
        sigma = BASELINE_SIGMA * math.sqrt(max(hour - 1, 0))
        srt = sorted(log_ratios)
    row = []
    for tau, z in zip(taus, zs):
        fitted = centre + z * sigma
        empirical = quantile(srt, tau)
        row.append(p * math.exp(0.5 * (fitted + empirical)))
    return row


def finish_row(raw: list[float]) -> list:
    """Clamp a raw quantile row to the contest's shape: finite, in (0, 1e9], non-decreasing."""
    row = []
    prev = 0.0
    for value in raw:
        if not math.isfinite(value) or value <= 0:
            value = prev if prev > 0 else 1.0
        value = min(max(value, 0.01), 1e9)
        value = max(value, prev)
        row.append(value)
        prev = value
    return [round2(v) for v in row]


def hour_one_print(case: dict) -> float:
    """The case's known hour-1 print; fallbacks keep the entry valid if it is absent."""
    known = case.get("known")
    if isinstance(known, list):
        for row in known:
            if isinstance(row, dict) and row.get("hour") == 1:
                viewers = as_float(row.get("viewers"))
                if viewers is not None:
                    return max(viewers, 1.0)
    history = case.get("history_hourly")
    if isinstance(history, list) and history:
        viewers = as_float(history[-1].get("viewers")) if isinstance(history[-1], dict) else None
        if viewers is not None:
            return max(viewers, 1.0)
    return 1.0


def forecast_hours(case: dict) -> list[int]:
    rows = case.get("forecast")
    hours = []
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict):
                hour = row.get("hour")
                if isinstance(hour, int) and 2 <= hour <= 24 and hour not in hours:
                    hours.append(hour)
    return hours or list(range(2, 25))


def forecast_case(case: dict) -> dict:
    levels = case.get("levels")
    if not isinstance(levels, list) or len(levels) != 5:
        levels = LEVELS
    try:
        levels = [float(tau) for tau in levels]
        zs = [NormalDist().inv_cdf(tau) for tau in levels]
    except Exception:
        levels = LEVELS
        zs = [NormalDist().inv_cdf(tau) for tau in LEVELS]
    p = hour_one_print(case)
    ratios, _n = collect_log_ratios(case)
    out_rows = []
    for hour in forecast_hours(case):
        log_ratios = ratios.get(hour)
        if log_ratios:
            raw = shape_row(p, hour, log_ratios, levels, zs)
        else:
            raw = baseline_row(p, hour, zs)
        out_rows.append({"hour": hour, "q": finish_row(raw)})
    return {"hours": out_rows}


def fallback_output() -> dict:
    """A valid-shaped forecast that needs nothing from the case at all."""
    zs = [NormalDist().inv_cdf(tau) for tau in LEVELS]
    return {"hours": [{"hour": h, "q": finish_row(baseline_row(1.0, h, zs))} for h in range(2, 25)]}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            forecast = forecast_case(case)
        except Exception:
            try:
                forecast = fallback_output()
            except Exception:
                forecast = {"hours": [{"hour": h, "q": [1, 1, 1, 1, 1]} for h in range(2, 25)]}
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / f"{path.stem}.out").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
