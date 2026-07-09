# Data

CSV files and the SQLite database are gitignored. Regenerate them as follows:

## Price data

```bash
# Download ETF and equity prices from Yahoo Finance (requires internet)
python -m src.download --asset-types equities etfs

# Incremental update (only new dates)
python -m src.download --incremental
```

NB: price CSVs and forecasts are regenerable; the **results tables** in
`portfolio.db` (`optimisation_runs`, `portfolio_holdings`, `backtest_sessions`,
`backtest_results`) are **not** — they encode random seeds and compute-hours.

## Forecasts

```bash
# Generate ARIMA expected returns and GARCH variances
python -m src.forecast
```

Outputs: `expected_returns.csv`, `variances.csv`

## Database

```bash
# Create empty database with schema
python -m src.db

# Import existing CSVs into the database
python -m src.db migrate
```

## Legacy files

- `time_series_20251016_113257.csv` -- InvestNow NZ managed fund prices. No automated download script; obtain manually if needed.
