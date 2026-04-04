# Data

CSV files and the SQLite database are gitignored. Regenerate them as follows:

## Price data

```bash
# Download ETF and equity prices from Yahoo Finance (requires internet)
python -m src.download_data --asset-types equities etfs

# Incremental update (only new dates)
python -m src.download_data --incremental
```

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
