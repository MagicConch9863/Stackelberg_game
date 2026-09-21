# Soft-SOC rebuild plan

## 1. New formulation

For BESS `i` with capacity `C_i`, the physical feasibility region is hard:

```text
0 <= E_i,k <= C_i
```

The preferred operating region is soft:

```text
E_i,k + v_i,k^- >= 0.20 C_i
E_i,k - v_i,k^+ <= 0.80 C_i
v_i,k^-, v_i,k^+ >= 0
```

The follower objective contains the piecewise-linear penalty

```text
rho_SOC * sum_k (v_i,k^- + v_i,k^+).
```

Thus the penalty is zero between 20% and 80%, grows linearly outside that
band, and states below 0% or above 100% are infeasible.  No day-ahead SOC
trajectory, terminal SOC corridor, day boundary, or midnight exception is
part of the new model.

## 2. Code layers

1. `phase4/soc_soft_band.py` is the single source of truth for the hard and
   soft SOC limits.
2. S2 and the direct BESS-MPC use explicit slack variables.
3. S3 and the rolling Stackelberg-MPC include the same slacks in the follower
   LP and therefore in its KKT reformulation.
4. S4 uses the same soft band so that the centralized comparison remains
   physically and methodologically comparable.
5. New results use formulation version `soft_soc_20_80_v1` and separate
   `results_soft_soc` folders.  Legacy caches and SOC-reference results are
   never valid inputs.

## 3. Rebuild sequence

1. **Model tests:** verify zero penalty at 50%, the correct linear violation
   at 10% and 90%, and infeasibility outside 0–100%.
2. **One-day smoke test:** run S1–S4 and T1–T6 for one day.  Inspect SOC,
   slacks, prices, power balance, solver status and KKT residuals.
3. **Boundary test:** run at least two continuous days and confirm there is no
   discontinuity tied to 00:00.  The same SOC rule must hold at every step.
4. **Four-day rebuild:** recreate all method and forecast comparisons from
   scratch.  Do not copy any old `comparison_*days` values.
5. **Ten-day rebuild:** only after the one-, two- and four-day checks pass,
   recompute T5-IID and T6-AR(1).
6. **Reporting:** regenerate plots, tables, paper text and presentation slides
   only from versioned soft-SOC results.

## 4. Required validation columns

Every new result set should record:

- formulation version;
- `rho_SOC`, lower and upper soft fractions;
- maximum physical-bound violation;
- total and maximum lower/upper soft-band violation;
- solver status, MIP gap and runtime per MPC update;
- power-balance and KKT residual checks.

## 5. Runtime evaluation

For a fixed 4-hour horizon, the number of MPC updates is `96 * D`, so the
engineering baseline for cumulative runtime is

```text
T(D) = a + b D.
```

Only runs made with `soft_soc_20_80_v1` on the same computer are fitted
together.  TSF points determine the blue linear fit; main-workstation points
are shown separately in green.  This empirical linear baseline does not imply
that the worst-case complexity of each MILP is linear.

## 6. Legacy status

All SOC-reference, midnight-fix, old four-day and old ten-day result folders
are historical only.  They may be retained for audit, but must not be cited,
plotted, averaged or used to fit runtime for the new formulation.
