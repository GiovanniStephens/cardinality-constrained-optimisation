# Data Refresh Runbook

How to bring `data/portfolio.db` price data up to date — reliably, and with
proof that the tickers you actually care about landed. Written after a June 2026
incident where a refresh *looked* successful but had silently skipped `SPY`,
`VOO`, `AGG` and other blue-chips for two months.

## TL;DR

```bash
make refresh          # full-universe refresh (hours) + auto-validate + health check
make health-check     # verify the liquid core is current & active (no download)
make retry-core       # recovery: force-refresh the core watchlist, bypass cache
```

A refresh is **not done** until `make health-check` shows `✅ all core tickers
current and active`. Trust the health check over the "Totals: X saved" summary.

## The two modes

| Command | What it does | When | Time |
|---|---|---|---|
| `make refresh` | Full universe (~42k tickers, all equities+ETFs, full history) | Routine; the canonical refresh | **Hours** (Yahoo throttles the proxy) |
| `make refresh-incremental` | Only dates newer than the latest already in the DB | Quick catch-up between full runs | Faster, but still touches the whole universe |

Both now **auto-run data-quality on production after promotion** and a
**health check** at the end — no manual `python -m src.data_quality` step.

## What the pipeline does (and where it can fail)

`download → stage → validate (staging) → promote → validate (production) → health check`

1. **Build ticker list** from FinanceDatabase (or `--from-csv`).
2. **Skip known-bad tickers** — *see the cache gotcha below*.
3. **Download** into a temporary `staging_*.db` (checkpointed).
4. **Validate staging** — on incremental runs `min_history` is skipped (staging
   only holds the new rows).
5. **Promote** staging → production (`INSERT OR REPLACE`, merges new dates).
6. **Validate production** (automatic) — sets `excluded` flags correctly on the
   merged data (clears stale flags, applies `min_history`).
7. **Health check** — asserts every core ticker (`data/core_etfs.csv`) is
   current and active; warns loudly with a recovery command if not.

## ⚠️ The bad-ticker cache (the thing that bit us)

Failed downloads are recorded in the `known_bad_tickers` table and **skipped on
every subsequent run**. A transient Yahoo failure can therefore blacklist a good
ticker. Hardening now in place (`src/db/bad_tickers.py`, `src/config.py`):

- **Protected watchlist** — tickers in `data/core_etfs.csv` are *never* cached or
  skipped. Blue-chips can't be blacklisted.
- **TTL self-heal** — entries expire after `PIPELINE_BAD_TICKER_TTL_DAYS` (30) and
  are retried. Legacy entries heal off `last_failed + TTL`.
- **Higher threshold** — a ticker must fail `PIPELINE_BAD_CACHE_MIN_FAILURES` (3)
  times before it's skipped; one blip no longer blacklists.

Inspect / clear the cache:

```bash
sqlite3 data/portfolio.db "SELECT COUNT(*) FROM known_bad_tickers;"
sqlite3 data/portfolio.db "SELECT * FROM known_bad_tickers WHERE symbol='SPY';"
python -c "from src import db; from src.db.bad_tickers import clear_known_bad_tickers as c; \
  conn=db.get_connection(); c(conn,'US'); conn.close()"   # nuke and start fresh
```

To retry tickers regardless of the cache, add `--ignore-bad-cache`.

## Reading the output

- **Per-asset summary:** `etf: status=promoted, saved=9203, failed=2040 tickers`.
  A low save rate is normal — most of the ~36k ETF universe is delisted/foreign
  tickers Yahoo won't serve.
- **Health check** (the one that matters):
  ```
  HEALTH CHECK (US)
    latest price date: 2026-06-15
    active tickers:    7520
    ✅ all 87 core tickers current and active
  ```
  If instead you see `⚠️ N/87 core tickers are STALE or MISSING`, run the printed
  recovery command (`make retry-core`).
- **Manifest:** `data/manifest_<run_id>.json` — full per-run detail incl.
  `validation`, `post_promotion_validation`, failed batches.

## Troubleshooting

**`status=validation_failed`, nothing promoted.**
Staging had 0 active tickers. On a *full* run this means the data is genuinely
bad. On an *incremental* run this should no longer happen (min_history is
skipped); if it does, the new rows were all stale/frozen — investigate the
source. To promote a good staging DB manually:
```python
from src import db
from src.pipeline import promote_staging
conn = db.get_connection()                 # production
promote_staging('data/staging_<id>.db', conn, exchange='US')
conn.commit(); conn.close()
```
Then re-validate: `python -m src.data_quality --exchange US` (or `make health-check`).

**A core ticker is stale / excluded.**
```bash
make health-check          # see which, and why (stale / min_history / missing)
make retry-core            # force a full-history re-fetch of the watchlist
```

**Retry *all* the tickers that failed this run** (broader than the core):
```python
# dump the bad cache to a CSV, then re-download with --ignore-bad-cache
from src import db
conn = db.get_connection(); ex = db._get_exchange_id(conn, 'US')
syms = [r[0] for r in conn.execute(
    "SELECT symbol FROM known_bad_tickers WHERE exchange_id=?", (ex,))]
conn.close()
import csv
with open('data/failed.csv','w',newline='') as f:
    csv.writer(f).writerows([['Tickers'], *[[s] for s in syms]])
# then: python -m src.download --from-csv data/failed.csv --asset-type etf --ignore-bad-cache --workers 24
```

**It's been hours / throttled.** Expected. Yahoo/Akamai rate-limits the proxy;
the pipeline backs off (circuit breaker, 300s cooldowns, up to 600s inter-batch).
More workers (`WORKERS=32`) is capped at `PIPELINE_MAX_WORKERS` (24) and rarely
helps — throttling, not worker count, is the wall.

**Validation warns about "phantom dates".** The July 2026 incident: for years,
`promote_staging` asked `load_prices` for no forward-fill via `ffill_limit=None`,
but pandas reads `limit=None` as *unlimited* fill — so every promotion stamped
each US ticker's previous close onto its 200-ticker chunk-mates' trading days
(Tel Aviv Sundays, Asian/European sessions on NYSE holidays). ~921k junk rows
(all verified previous-close duplicates) accumulated; on the polluted union
date index SPY itself read 88% coverage and silently fell out of the backtest
frame. Fixed at the root (`load_prices` now treats falsy `ffill_limit` as no
fill), guarded at write time (`save_prices` drops weekend-dated rows for
US-listed symbols), detected on every validation run, and purged 2026-07-10.
If the warning ever fires again:
```bash
python -m src.db purge-phantom-rows --dry-run   # inspect counts first
python -m src.db purge-phantom-rows             # backs up, then deletes
```
And investigate what wrote the rows — the write guard means a recurrence
implies a NEW writer path bypassing `save_prices`' weekend check (holiday rows
especially: those are only caught by detection/purge, not at write time).

## Key config (`src/config.py`)

| Constant | Default | Meaning |
|---|---|---|
| `PIPELINE_MAX_WORKERS` | 24 | hard cap on `--workers` |
| `PIPELINE_BAD_CACHE_MIN_FAILURES` | 3 | failures before a ticker is skipped |
| `PIPELINE_BAD_TICKER_TTL_DAYS` | 30 | bad-cache entry expiry (self-heal) |
| `PIPELINE_PROTECTED_TICKERS_CSV` | `data/core_etfs.csv` | never-skip watchlist |
| `MIN_HISTORY_DAYS` | 1260 | ≥5yr history to be active |
| `MAX_STALENESS_DAYS` | 30 | last trade must be this recent |
