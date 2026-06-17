# Cell Failure Analysis Practice Case: Cell A-17

## Background

Cell A-17 is a graphite/NMC532 pouch cell (3.0 Ah nameplate) from a cycle-life
qualification lot. It was cycled at 1C/1C between 2.80 V and 4.20 V at room
temperature on a research-grade cycler. Every 5th cycle, the protocol
inserted a low-rate (C/20) reference charge/discharge to suppress kinetic
(IR/polarization) artifacts in the voltage curve, making these "checkup"
cycles suitable for differential voltage analysis (DVA).

You are handed the data below as if it just came off the cycler. Your job is
to work the problem the way you would in a real post-mortem screen: characterize
the fade, build the DVA, identify which degradation mode(s) are present, and
make a capacity-fade attribution. No mechanism is stated anywhere in this
document — that's the point.

## What you have

**`cycle_summary.csv`** — every cycle (0–80), three columns:
- `cycle_number`
- `discharge_capacity_mAh` (1C discharge capacity, as logged each cycle)
- `coulombic_efficiency` (per-cycle CE)

**`checkup_discharge_curves.csv`** — full C/20 voltage-vs-capacity traces at
17 checkup cycles (0, 5, 10, ..., 80), three columns:
- `cycle_number`
- `capacity_mAh`
- `voltage_V`

**`discharge_curves_overview.png`** — a quick-look plot of 9 of the 17 checkup
curves overlaid, included only as a sanity-check visual. Don't use it as your
analysis — build your own DVA from the CSV.

## Data realism notes (read before you panic about "noise")

- Capacity values include realistic cycler repeatability noise (~0.05–0.1%
  cycle-to-cycle scatter) — don't over-interpret point-to-point jitter in the
  fade curve as a real event unless it's a sustained trend.
- Checkup voltage curves include small voltage noise (~0.1 mV level, consistent
  with a precision research cycler) and have been resampled onto a 0.5 mAh
  capacity grid. You will need to smooth before differentiating (your call on
  method — Savitzky-Golay, a fitted spline, binned/windowed central
  differences, etc. — but make a defensible choice and say what you used and why).
- The first ~1.5% and last ~1.5% of each checkup curve are mildly affected by
  the voltage-cutoff turnaround and are noisier in dV/dQ than the bulk —
  this is a real cycler artifact, not a bug in this dataset, but you may want
  to truncate or downweight those edges in your derivative.

## Suggested workflow

1. Plot capacity fade and CE vs. cycle number first. Before touching DVA, ask
   what the fade *shape* (not just the endpoint number) is already telling
   you — is it linear, accelerating, does it have a knee, is there a delayed
   onset?
2. Build dV/dQ for each checkup cycle. Identify the resolvable peaks/features
   in the fresh-cell (cycle 0) curve and track each one's position, height,
   and area across cycles.
3. Classify what you're seeing in standard DVA-fitting language: are peaks
   shifting in capacity (consistent with LLI), shrinking/growing in height or
   area without shifting (consistent with LAM at one electrode), or some
   combination? Is the effect uniform across all peaks, or concentrated in a
   subset (which would point at which electrode is implicated)?
4. Cross-check your DVA-based read against the raw capacity fade and CE
   trends — do they tell a consistent story?
5. Write a one-paragraph attribution: which degradation mode(s) dominate, what's
   your evidence, and what's your confidence/what would you want to confirm
   with destructive analysis (e.g. reference electrode rebuild, post-mortem
   SEM/XRD) if this were a real cell.

## What I'll do when you're ready

When you've got your DVA built and a candidate diagnosis, bring it back and
I'll tell you the ground-truth mechanism I encoded and we can go through where
your read matched, where it didn't, and why — including any places the
"real" signature was more subtle or ambiguous than textbook examples make it
look (which is intentional; this case is not a pure single-mode textbook
example).
