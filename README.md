# gta6-twitch-ccu-forecast-01

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/cubinmark/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/cubinmark/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

The entry for make.fun's “Forecast GTA VI's launch-day Twitch viewers hour by hour” bounty. `main.py` reads each case in the input directory and writes five quantiles (5/25/50/75/95%) for hours 2–24, learning the launch-day curve from the case's comparable launches, filtered to the rolling midnight-unlock regime GTA VI follows. Leave-one-out loss 0.2537 vs the house baseline's 0.2965 on comparable launches.

## Run it

```sh
MAKEFUN_INPUT_DIR=<folder with .in case files> MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=dnrr099a&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>

## Model notes

Forecasts GTA VI's launch-day Twitch viewers, hour by hour (hours 2–24 UTC of
2026-11-19), as five quantiles per hour, from the case's hour-1 print and its
comparables.

## Model

The house baseline holds every hour flat at hour 1. This entry instead learns
the release-day curve's shape from the case's own comparables:

1. Each comparable's release day is aligned to UTC hours 1–24 from its
   `day_start`, giving ratios `viewers_h / viewers_1`.
2. Only comparables whose hour 1 was already a real launch hour — at least 5%
   of their release-day peak, the signature of a rolling midnight-local unlock
   like GTA VI's — shape the forecast. (Launches that sat near zero until a
   single global unlock hour would teach the wrong shape.) Fallbacks widen the
   set, then fall back to the baseline formula, if too few qualify.
3. Per hour, the forecast's log-quantiles average two views of the log-ratios:
   a normal fit (median centre, IQR-based sigma) and the empirical quantiles —
   the average keeps the centre smooth and the tails honest.
4. The case's hour-1 print `p` anchors the level: `q_tau(h) = p * exp(lq)`.

## Validation

Leave-one-out over the 37 usable comparables of the practice data (each
comparable forecast from its own hour 1 with the model trained on the other
36), scored with the contest's pinball loss on the log scale:

| pseudo-cases | this model | house baseline formula |
| --- | --- | --- |
| hour 1 ≥ 10% of day peak (GTA VI's regime) | ≈ 0.253 | ≈ 0.297 |
| all 37 | ≈ 0.834 | ≈ 0.930 |

## Running

The grader runs `python3 -P -s /bot/main.py` with one case in `/input`.
Locally:

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Standard library only, no network, deterministic output.
