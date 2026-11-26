# Whole-house thermostat policy

The bedroom and living-room sensors are independent. The study borrows the living-room measurement. Use each virtual thermostat's active target, with an ideal band of ±0.5 °C and a comfort band of ±0.7 °C. Do not average the zones.

## Action timing

- Preserve the normal 1 °C step. The existing 2 °C option still needs an error of at least 2 °C and a non-recovering trend. Keep device limits, 15 minutes minimum run, five minutes confirmed low draw after stopping, and existing 60-minute reversal protection.
- Increasing output waits 10 minutes after the previous whole-house action and for independent feedback. A permitted local start waits until the measured excursion reaches 20 minutes. It also requires a non-recovering trend or two actual out-of-band reports at least 10 minutes apart. The excursion, minimum off time, and the prior strengthening observation run in parallel. If a prior action was a stop, a new relevant report is required after the affected stop and, with all units off, after the last house stop. Stale data, unavailable feedback, contradictory demand, pending receipts or insufficient low-power evidence can extend the wait. No fixed maximum applies to a safety hold. Once conditions are met the next 20-second cycle may select one device.
- A well-served or comfortable lead does not have to reach a setpoint limit before another independently measured zone can start. A newly started unit becomes the only candidate for that demand until fresh response is available. Near the opposite ideal boundary, prefer the directly measured zone, skip the study as auxiliary output, and limit local increases to one degree. If the two independent measurements disagree across the comfort band, only reductions of adverse output are allowed.
- A reduction may follow after three minutes and a real new temperature report. Normal trend uses distinct post-change reports, typically three in 15 minutes. Sparse fallback uses two reports at least 10 minutes apart in 32 minutes. Two reports at least four minutes apart with at least 0.25 °C of movement and a recent latest report also qualify. Do not carry a trend across a target, setpoint, mode, temperature sensor, or actuation change.
- High-power easing is independent from projected overshoot. It requires temperature across the active mode's ideal line, the room's high-power duration (default three minutes), matched power source and a fresh reading above the current calibrated high threshold, and no credible recovery. A unit that has already reduced output can exit once meaningful output and adverse feedback remain and the direction is no longer needed. The gate preserves minimum runtime and a restart reference. Dormant labels alone do not trigger a controlled shutdown unless they block a sustained opposite demand.
- After a mild excursion (at most 1 °C), the initial setting is approached conservatively using the current target, the learned initial setting and the last comparable acknowledged setting or companion memory when available. A remembered setting weaker than the fallback can be retained. The mode and hardware limits still apply. An irrelevant unsent initial setting drops out of the batch once fresh data verifies the change in demand. Accepted and ambiguous commands remain in the journal.

## Learning and audit

Two independent, early/late temperature reports must span the 20-minute episode, with no gap over 16 minutes and coverage within five minutes of both ends. A qualifying target is constant, every observation stays within ±0.5 °C of it, net drift stays within 0.2 °C, and total spread stays within 0.3 °C. Power evidence must span the episode without a report gap, source change or reclassification backfill. New quality marks are in an additive SQLite table. Earlier samples remain in the database and are excluded from the updated learner. The study's episodes are recorded and excluded from its applied compensation.

The pilot writes privately mounted JSON Lines audit events with up to eight 4 MiB files. Cycles record demand, lead, actual reading ages, trend validity, candidates, gate refusals, selections and journal command time. Receipt records distinguish proxy echo and transport outcomes. Exception stacks contain basename, function, and line number, with no raw payload or exception text. The startup marker includes the installed Git commit, code hash and static effective configuration hash. The shadow summary records observed comfort, excursion duration and recovery, covered energy and proxy actuation count. Common kWh is counted only for complete minutes with qualified power evidence from all three units. Proxy mode changes and measured load are separate observations.

## Reproduce and release

```sh
export ROOM_CLIMATE_ROOMS_FILE="$PWD/home-assistant/shadow/rooms.example.json"
python3 -B -m unittest discover -s work -p 'test_*.py'
python3 -B work/replay_acceptance.py /path/to/sanitized-acceptance.zip
```

The replay rebuilds assessments through the actual gate and journal against fixed observations from the archive, with simulated HA calls. Real 20-second reasons from the earlier build were never archived. The old reconstructed 14:41–14:52 interval includes the 20-minute excursion dwell, release of opposing mode, the post-14:40 stop 10-minute house observation and dense-trend requirement (still waiting at 14:51; eligible at 14:53). The prior living-room mode was already cool, so the old same-direction 20-minute dwell applied. The new replay can select the study stop at 14:34 and living-room cooling at 14:48. At 14:54 it retains the companion's 26 °C cooling setting. At 15:07 it selects trend easing, and at 15:18 it can stop cooling after continued adverse feedback. Each slice reuses the historical inputs, so the slices cannot be concatenated into actual temperatures, causal effects or a kWh forecast.

On the existing production host, deploy only the commit already pushed to `main`. The release script carries across the local mapping and any existing provisional off-power references. It quiesces the two sidecars, checks for outstanding journal records, archives code/config and read-only journal/SQLite evidence, and restarts the sidecars. For rollback run the same script with the backup path printed during release. Rollback restores the previous compatible code and config only.

```sh
python3 work/release_local.py --production /path/to/home-assistant --commit FULL_SHA
python3 work/release_local.py --production /path/to/home-assistant --rollback /path/to/release-backups/RELEASE
```

After a release confirm the startup commit and hashes in `pilot-data/live-audit.jsonl`, fresh cycles, current locks and any naturally dispatched relay/proxy receipt pairs. Leave normal operation running across day and night to assess independent comfort and metered kWh under comparable coverage and weather.
