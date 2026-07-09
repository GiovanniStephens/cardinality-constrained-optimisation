# Quarterly Rebalance Runbook

How to produce and sanity-check the production portfolio — reliably, with the
failure modes named. Written after a July 2026 incident chain where a skipped
manual step (name backfill after a symbol-directory ingest) made the category
caps mathematically infeasible, SLSQP silently fell back to equal weights, and
a cap-breaching book printed as if it were clean.

## TL;DR

```bash
make health-check                     # 1. data is current (see DATA_REFRESH.md)
python -m src.db backfill-names       # 2. only needed after a directory ingest
python -m src.db backfill-volume      # 3. keep ADV coverage current
python run_rebalance.py --portfolio-value <NZD/USD amount>   # 4. ~6-8 min
# 5. read the DEPLOYABLE BOOK block — trade ONLY if there is no
#    "!! DO NOT TRADE AS-IS" banner — then place limit orders manually.
```

The defaults carry the **full validated config** — no flags needed for a
standard quarter: category caps (Unknown ≤15%, Inverse ≤10%, …), 12% return
floor, 0.50 beta floor vs SPY, 5–12% per-holding weights, 10–15 holdings,
$500k ADV liquidity floor, SMH must-have, 75/25 equity/managed-futures split.

## Preconditions (in order)

1. **Data is fresh and verified**: `make refresh` has completed recently and
   `make health-check` shows `✅ all core tickers current` — see
   `docs/DATA_REFRESH.md`. Do not rebalance on stale prices.
2. **Names are populated**: after ANY new-symbol ingest run
   `python -m src.db backfill-names`. Unnamed tickers all classify as
   `Unknown`; enough of them in one basket makes the caps infeasible (the
   July 2026 incident). The run now pre-checks this arithmetic and logs the
   binding group at ERROR level — but backfilling names first avoids the
   problem instead of diagnosing it.
3. **Volume is populated**: `python -m src.db backfill-volume` so the ADV
   liquidity filter has current data.

## The run

```bash
python run_rebalance.py --portfolio-value 100000
```

~6–8 minutes: ~3 min data load, ~2 min GA (600s budget, usually converges
early), ~1 min weighting + report. Every run saves each method's book AND the
deployable blend to the DB with the full parameter record (CLI args, category
caps, git commit, resolved seed, universe hash) — note the printed `run_id`s.

Useful overrides (all recorded in the run params):

| Flag | Default | Purpose |
|---|---|---|
| `--min-return` | 0.12 | Return floor; `<= 0` disables (pure max-Sharpe) |
| `--min-beta` | 0.50 | Beta floor vs SPY; `<= 0` disables |
| `--max-weight` | 0.12 | Per-holding ceiling (`config.REBALANCE_MAX_WEIGHT`) |
| `--must-have` | SMH | Forced holdings, comma-separated; `''` disables |
| `--sleeve-alpha` / `--no-sleeve` | 0.25 | Managed-futures capital split |
| `--seed` | random (resolved + recorded) | Re-run a previous book's seed |

## Post-run sanity checklist — before placing any order

Read the report bottom-up:

- [ ] **No `!! DO NOT TRADE AS-IS` banner** under DEPLOYABLE BOOK. The banner
      lists exactly what is wrong (SLSQP fallback, cap breach, return/beta
      floor miss, method substitution). If it appears, fix the cause and
      re-run — do not hand-adjust weights.
- [ ] **Notes column is clean** for the recommended method: no
      `SLSQP FALLBACK (1/N)`, no `N cap breach`, no `return<12% target`, no
      `beta<0.50 target`. (`IS>1.5 suspect` is informational — see below.)
- [ ] **Holdings count** is within 10–15 and no single weight exceeds 12%
      (equity book, pre-split).
- [ ] **Beta ≈ 0.50+** on the recommended row (the floor usually binds at
      0.50 exactly; materially above is fine, below means it failed).
- [ ] **The book passes the smell test**: ~6 real equity legs carrying the
      beta, the rest diversifiers; nothing you can't explain.
- [ ] **cc selection scan line** in the log shows a healthy compliant count
      (`cc selection scan: N candidates weighted, M compliant`); `0 compliant`
      means the constraints are fighting the GA — investigate before trading.

**Interpreting the Sharpe numbers**: the IS Sharpe is biased upward by
construction; the OOS column (50% haircut × Harvey-Liu) is the honest
estimate, and the realistic ceiling for this book is ~1.0–1.2 (CLAUDE.md,
"Sharpe Ratio Overfitting"). An IS Sharpe *above ~3* means something broke —
historically it meant the optimiser found a constraint hole (see the
Goodhart/beta-floor arc, Lesson 13).

## Orders

- Trade the **DEPLOYABLE BOOK** table (equity legs already scaled by 0.75,
  sleeve ETFs at ~8.3% each) — not the raw per-method tables.
- All names have cleared the $500k/day ADV floor; still use **limit orders
  near mid**, and work younger/thinner funds patiently.
- Record the deployable `run_id` printed at the end (also in
  `optimisation_runs` as `rebalance_deployable`) — next quarter's comparison
  starts from it.

## OOS gate (occasional, not quarterly)

`python backtest_rebalance.py` walk-forwards the *production config*
(quarterly re-optimise → hold) across history and reports the realised OOS
Sharpe vs SPY. Run it after any material config change (caps, floors, weight
ceiling) — hours, not minutes; use `--max-windows` for a smoke run.
