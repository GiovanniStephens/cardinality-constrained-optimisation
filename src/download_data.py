"""
Builds a security universe from FinanceDatabase and downloads price data
from Yahoo Finance.

Supports equities, ETFs, and funds. Can also load tickers from a local CSV
for working with previously scraped lists.
"""

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

import financedatabase as fd
import pandas as pd
import yfinance as yf
from tqdm import tqdm

from src.exceptions import DownloadError, ValidationError

logger = logging.getLogger(__name__)

# Maps FinanceDatabase asset type labels to DB-canonical names.
# DB migration uses 'stock' for equities (see db._migrate_ticker_list).
ASSET_TYPE_MAP = {'equity': 'stock', 'etf': 'etf', 'fund': 'fund'}


# ---------------------------------------------------------------------------
# Universe building (from FinanceDatabase)
# ---------------------------------------------------------------------------

def _filter_by_exchange(df, exchanges):
    """Post-filter a FinanceDatabase result by exchange column.

    The library's select() does not accept 'exchange' as a keyword, so we
    filter the returned DataFrame manually.
    """
    if exchanges and 'exchange' in df.columns:
        if isinstance(exchanges, str):
            exchanges = [exchanges]
        df = df[df['exchange'].isin(exchanges)]
    return df


def get_equities(countries=None, sectors=None, industries=None,
                 exchanges=None, market_caps=None) -> pd.DataFrame:
    """
    Retrieves equity tickers from FinanceDatabase.

    :param countries: country or list of countries to filter by.
    :param sectors: sector or list of sectors to filter by.
    :param industries: industry or list of industries to filter by.
    :param exchanges: exchange or list of exchanges to filter by (post-filter).
    :param market_caps: market cap category or list to filter by (post-filter).
    :returns: DataFrame with ticker symbols and metadata.
    """
    equities = fd.Equities()
    kwargs = {}
    if countries:
        kwargs['country'] = countries
    if sectors:
        kwargs['sector'] = sectors
    if industries:
        kwargs['industry'] = industries
    df = _filter_by_exchange(equities.select(**kwargs), exchanges)
    if market_caps and 'market_cap' in df.columns:
        if isinstance(market_caps, str):
            market_caps = [market_caps]
        df = df[df['market_cap'].isin(market_caps)]
    return df


def get_etfs(category_groups=None, categories=None, families=None,
             exchanges=None) -> pd.DataFrame:
    """
    Retrieves ETF tickers from FinanceDatabase.

    :param category_groups: category group or list to filter by.
    :param categories: category or list to filter by.
    :param families: ETF family/provider or list to filter by.
    :param exchanges: exchange or list of exchanges to filter by.
    :returns: DataFrame with ticker symbols and metadata.
    """
    etfs = fd.ETFs()
    kwargs = {}
    if category_groups:
        kwargs['category_group'] = category_groups
    if categories:
        kwargs['category'] = categories
    if families:
        kwargs['family'] = families
    return _filter_by_exchange(etfs.select(**kwargs), exchanges)


def get_funds(category_groups=None, categories=None, families=None,
              exchanges=None) -> pd.DataFrame:
    """
    Retrieves fund tickers from FinanceDatabase.

    :param category_groups: category group or list to filter by.
    :param categories: category or list to filter by.
    :param families: fund family/provider or list to filter by.
    :param exchanges: exchange or list of exchanges to filter by.
    :returns: DataFrame with ticker symbols and metadata.
    """
    funds = fd.Funds()
    kwargs = {}
    if category_groups:
        kwargs['category_group'] = category_groups
    if categories:
        kwargs['category'] = categories
    if families:
        kwargs['family'] = families
    return _filter_by_exchange(funds.select(**kwargs), exchanges)


def build_security_universe(asset_types=None, countries=None, sectors=None,
                            industries=None, exchanges=None,
                            market_caps=None, etf_categories=None,
                            etf_category_groups=None) -> pd.DataFrame:
    """
    Builds a combined universe of securities from multiple asset types.

    :param asset_types: list of asset types to include.
        Options: 'equities', 'etfs', 'funds'. Defaults to all three.
    :param countries: country filter (applies to equities).
    :param sectors: sector filter (applies to equities).
    :param industries: industry filter (applies to equities).
    :param exchanges: exchange filter (applies to equities only).
    :param market_caps: market cap filter (applies to equities).
    :param etf_categories: category filter (applies to ETFs).
    :param etf_category_groups: category group filter (applies to ETFs).
    :returns: DataFrame with columns ['Tickers', 'Name', 'Country', 'AssetType'].
    """
    if asset_types is None:
        asset_types = ['equities', 'etfs', 'funds']

    all_securities = []

    if 'equities' in asset_types:
        equities = get_equities(countries=countries, sectors=sectors,
                                industries=industries, exchanges=exchanges,
                                market_caps=market_caps)
        if not equities.empty:
            eq_df = pd.DataFrame({
                'Tickers': equities.index,
                'Name': equities['name'] if 'name' in equities.columns else '',
                'Country': equities['country'] if 'country' in equities.columns else '',
                'AssetType': 'equity'
            })
            all_securities.append(eq_df)

    if 'etfs' in asset_types:
        etfs = get_etfs(category_groups=etf_category_groups,
                        categories=etf_categories)
        if not etfs.empty:
            etf_df = pd.DataFrame({
                'Tickers': etfs.index,
                'Name': etfs['name'] if 'name' in etfs.columns else '',
                'Country': '',
                'AssetType': 'etf'
            })
            all_securities.append(etf_df)

    if 'funds' in asset_types:
        funds = get_funds()
        if not funds.empty:
            fund_df = pd.DataFrame({
                'Tickers': funds.index,
                'Name': funds['name'] if 'name' in funds.columns else '',
                'Country': '',
                'AssetType': 'fund'
            })
            all_securities.append(fund_df)

    if not all_securities:
        return pd.DataFrame(columns=['Tickers', 'Name', 'Country', 'AssetType'])

    combined = pd.concat(all_securities, ignore_index=True)
    combined = combined.drop_duplicates(subset='Tickers')
    return combined


# ---------------------------------------------------------------------------
# Ticker loading from CSV (for previously scraped lists)
# ---------------------------------------------------------------------------

def load_tickers(filename: str, ticker_column: str = 'Tickers') -> pd.DataFrame:
    """
    Loads the list of tickers from a local CSV file.

    :param filename: path to CSV file containing tickers.
    :param ticker_column: name of the column containing ticker symbols.
    :returns: DataFrame with at least the ticker column.
    """
    tickers = pd.read_csv(filename)
    if tickers.empty:
        raise ValueError(f"Ticker list file '{filename}' is empty.")
    if ticker_column not in tickers.columns:
        raise ValueError(
            f"Column '{ticker_column}' not found in {filename}. "
            f"Available columns: {list(tickers.columns)}"
        )
    return tickers


# ---------------------------------------------------------------------------
# Price downloading
# ---------------------------------------------------------------------------

def _download_batch(tickers, start, end):
    """Download a single batch from yfinance. Returns wide DataFrame or None."""
    tickers_str = " ".join(tickers)
    prices = yf.download(
        tickers_str, interval="1d", group_by="ticker", start=start, end=end,
    )
    # Validate returned structure (P5)
    if prices is None or not isinstance(prices, pd.DataFrame) or prices.empty:
        return None
    if not isinstance(prices.index, pd.DatetimeIndex):
        try:
            prices.index = pd.to_datetime(prices.index)
        except (ValueError, TypeError) as e:
            logger.warning("Batch returned non-date index type: %s (%s)", type(prices.index), e)
            return None

    batch_prices = {}
    skipped = []
    for ticker in tickers:
        try:
            if len(tickers) == 1:
                batch_prices[ticker] = prices["Close"].tolist()
            else:
                batch_prices[ticker] = prices[ticker]["Close"].tolist()
        except (KeyError, TypeError):
            skipped.append(ticker)
    # Log aggregated summary instead of per-ticker warnings (P7)
    if skipped:
        preview = ', '.join(skipped[:10])
        suffix = f'... (+{len(skipped) - 10} more)' if len(skipped) > 10 else ''
        logger.info("Batch: %d/%d tickers had no data: %s%s",
                     len(skipped), len(tickers), preview, suffix)
    if not batch_prices:
        return None
    return pd.DataFrame(batch_prices, index=prices.index)


def _download_batch_with_timeout(tickers, start, end, timeout_seconds):
    """Wrap _download_batch with a timeout. Returns None on timeout."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_download_batch, tickers, start, end)
        try:
            return future.result(timeout=timeout_seconds)
        except FuturesTimeout:
            logger.warning("Batch download timed out after %ds (%d tickers)",
                           timeout_seconds, len(tickers))
            return None


def download_data(
    tickers_df: pd.DataFrame,
    ticker_column: str = "Tickers",
    start: str = "2014-04-30",
    end: str = "2025-04-30",
    batch_size: int = 500,
) -> pd.DataFrame:
    """
    Downloads closing price data from Yahoo Finance for the given tickers.

    Processes tickers in batches to avoid timeouts with large ticker lists.

    :param tickers_df: DataFrame containing ticker symbols.
    :param ticker_column: name of the column containing ticker symbols.
    :param start: start date for price data.
    :param end: end date for price data.
    :param batch_size: number of tickers to download per batch.
    :returns: daily closing price data as a DataFrame.
    """
    all_tickers = tickers_df[ticker_column].tolist()
    batches = [
        all_tickers[i : i + batch_size]
        for i in range(0, len(all_tickers), batch_size)
    ]

    all_prices = {}
    for batch_num, batch in enumerate(batches, 1):
        logger.info("Downloading batch %d/%d (%d tickers)...",
                     batch_num, len(batches), len(batch))
        batch_df = None
        for attempt in range(1, 4):
            try:
                batch_df = _download_batch(batch, start, end)
                break
            except (ConnectionError, OSError, DownloadError) as e:
                if attempt == 3:
                    raise DownloadError(
                        f"Failed after 3 retries: {e}") from e
                logger.warning("Batch %d attempt %d failed: %s",
                               batch_num, attempt, e)
                time.sleep(2 ** attempt)
            except Exception as e:
                if attempt == 3:
                    raise DownloadError(
                        f"Failed after 3 retries (unexpected): {e}") from e
                logger.warning("Batch %d attempt %d failed (unexpected): %s",
                               batch_num, attempt, e)
                time.sleep(2 ** attempt)
        if batch_df is not None:
            for col in batch_df.columns:
                all_prices[col] = batch_df[col].tolist()

    if not all_prices:
        raise ValueError("No valid price data could be extracted for any ticker.")
    return pd.DataFrame(all_prices)


def download_and_save(
    tickers, conn, exchange, asset_type='etf',
    start='2014-04-30', end='2025-04-30',
    batch_size=500, null_threshold=0.9,
    names=None, countries=None, max_retries=3,
    on_batch_complete=None, on_batch_failed=None,
    rate_limit_delay=0.0, batch_timeout=None,
    circuit_breaker_threshold=None, max_rate_limit_delay=None,
):
    """
    Download prices in batches and persist each batch to the database immediately.

    This is the preferred path for large universes (1000+ tickers). Each batch is
    saved via upsert so the run can be safely interrupted and resumed.

    :param tickers: list of ticker symbol strings.
    :param conn: open sqlite3 connection (from db.get_connection()).
    :param exchange: DB exchange code ('US', 'NZX', 'ASX').
    :param asset_type: DB asset type ('etf', 'stock', 'fund').
    :param start: start date for price data.
    :param end: end date for price data.
    :param batch_size: number of tickers per yfinance request.
    :param null_threshold: fraction of non-null rows required to keep a ticker.
    :param names: optional dict {symbol: name_string}.
    :param countries: optional dict {symbol: country_string}.
    :param max_retries: retries per batch on download failure.
    :param on_batch_complete: optional callback(saved_tickers: list[str], batch_num: int)
        called after each successful batch save.
    :param on_batch_failed: optional callback(failed_tickers: list[str], batch_num: int)
        called when a batch exhausts all retries.
    :param rate_limit_delay: seconds to sleep between batches (default: 0).
    :param batch_timeout: seconds before a single batch download times out.
    :param circuit_breaker_threshold: consecutive failed batches before aborting.
    :param max_rate_limit_delay: maximum adaptive inter-batch delay (seconds).
    :returns: dict with keys total_tickers, saved_tickers, failed_batches,
        and circuit_breaker_tripped (bool).
    """
    from src import db, config

    # Apply config defaults
    if batch_timeout is None:
        batch_timeout = config.PIPELINE_BATCH_TIMEOUT
    if circuit_breaker_threshold is None:
        circuit_breaker_threshold = config.PIPELINE_CIRCUIT_BREAKER_THRESHOLD
    if max_rate_limit_delay is None:
        max_rate_limit_delay = config.PIPELINE_MAX_RATE_LIMIT_DELAY

    # Deduplicate ticker list (P6)
    seen = set()
    deduped = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            deduped.append(t)
    if len(deduped) < len(tickers):
        logger.info("Removed %d duplicate tickers", len(tickers) - len(deduped))
    tickers = deduped

    batches = [
        tickers[i : i + batch_size]
        for i in range(0, len(tickers), batch_size)
    ]
    total_saved = 0
    failed_batches = []
    consecutive_failures = 0
    circuit_breaker_tripped = False
    current_delay = rate_limit_delay
    t_start = time.time()

    use_tqdm = sys.stderr.isatty()
    batch_iter = enumerate(batches, 1)
    if use_tqdm:
        pbar = tqdm(total=len(batches), desc="Downloading", unit="batch",
                    file=sys.stderr)
    else:
        pbar = None

    for batch_num, batch in batch_iter:
        if not use_tqdm:
            logger.info("Batch %d/%d (%d tickers)...",
                        batch_num, len(batches), len(batch))
        batch_df = None
        hit_rate_limit = False
        for attempt in range(1, max_retries + 1):
            try:
                batch_df = _download_batch_with_timeout(
                    batch, start, end, batch_timeout)
                if batch_df is not None and not batch_df.empty:
                    break
                # Empty result likely means Yahoo silently rate-limited
                # (yfinance doesn't raise on throttle, just returns no data)
                if attempt < max_retries:
                    hit_rate_limit = True
                    backoff = 2 ** (attempt + 1)
                    logger.warning("Batch %d attempt %d/%d: no data returned, "
                                   "likely rate-limited. Backing off %ds",
                                   batch_num, attempt, max_retries, backoff)
                    time.sleep(backoff)
            except Exception as e:
                error_msg = str(e).lower()
                if '429' in error_msg or 'too many' in error_msg or 'rate' in error_msg:
                    hit_rate_limit = True
                    backoff = 2 ** (attempt + 2)
                    logger.warning("Rate limited on batch %d, backing off %ds",
                                   batch_num, backoff)
                    time.sleep(backoff)
                else:
                    logger.warning("Batch %d attempt %d/%d failed: %s",
                                   batch_num, attempt, max_retries, e)
                    logger.debug("Batch %d traceback:", batch_num, exc_info=True)
                    if attempt < max_retries:
                        time.sleep(2 ** attempt)

        if batch_df is None or batch_df.empty:
            logger.warning("Batch %d: no data returned, skipping.", batch_num)
            failed_batches.append({'batch_num': batch_num, 'tickers': list(batch)})
            consecutive_failures += 1
            if on_batch_failed:
                on_batch_failed(list(batch), batch_num)
            if pbar:
                pbar.update(1)

            # Circuit breaker (P1)
            if consecutive_failures >= circuit_breaker_threshold:
                logger.error(
                    "Circuit breaker tripped: %d consecutive batch failures. "
                    "Aborting download. Use --resume to retry later.",
                    consecutive_failures)
                circuit_breaker_tripped = True
                break
            continue

        # Reset circuit breaker on success
        consecutive_failures = 0

        # Adaptive rate limit: escalate after 429, decay after clean success (P3)
        if hit_rate_limit:
            current_delay = min(current_delay * 2, max_rate_limit_delay)
            logger.info("Rate limit hit, inter-batch delay increased to %.1fs",
                        current_delay)
        else:
            current_delay = max(rate_limit_delay, current_delay * 0.8)

        # Filter out tickers with too many nulls
        pre_filter_count = len(batch_df.columns)
        threshold = int(len(batch_df) * null_threshold)
        batch_df = batch_df.dropna(axis=1, thresh=threshold)
        dropped = pre_filter_count - len(batch_df.columns)
        if dropped:
            logger.info("Batch %d: dropped %d/%d tickers below %.0f%% coverage",
                        batch_num, dropped, pre_filter_count, null_threshold * 100)
        if batch_df.empty:
            if pbar:
                pbar.update(1)
            continue

        # Build per-batch metadata dicts
        batch_names = None
        if names:
            batch_names = {t: names[t] for t in batch_df.columns if t in names}
        batch_countries = None
        if countries:
            batch_countries = {t: countries[t] for t in batch_df.columns if t in countries}

        db.save_prices(conn, batch_df, exchange=exchange, asset_type=asset_type,
                       names=batch_names, countries=batch_countries)
        saved_tickers_list = list(batch_df.columns)
        total_saved += len(saved_tickers_list)

        # Enhanced progress reporting (P8)
        elapsed = time.time() - t_start
        if pbar:
            pbar.update(1)
            pbar.set_postfix(
                saved=total_saved,
                failed=len(failed_batches),
                rate=f"{total_saved / elapsed:.0f}/s" if elapsed > 0 else "N/A",
            )
        else:
            logger.info("Batch %d: saved %d tickers (total: %d)",
                        batch_num, len(saved_tickers_list), total_saved)

        if on_batch_complete:
            on_batch_complete(saved_tickers_list, batch_num)

        if current_delay > 0 and batch_num < len(batches):
            time.sleep(current_delay)

    if pbar:
        pbar.close()

    # Summary log (P7)
    if failed_batches:
        total_failed_tickers = sum(len(fb['tickers']) for fb in failed_batches)
        logger.warning("Download summary: %d/%d batches failed (%d tickers). "
                       "Use checkpoint to retry.",
                       len(failed_batches), len(batches), total_failed_tickers)

    return {
        'total_tickers': len(tickers),
        'saved_tickers': total_saved,
        'failed_batches': failed_batches,
        'circuit_breaker_tripped': circuit_breaker_tripped,
    }


def save_to_csv(prices: pd.DataFrame, filename: str) -> None:
    """
    Saves the given DataFrame to a CSV file.

    :param prices: daily price data.
    :param filename: name of the file to save the data to.
    """
    prices.to_csv(filename)
    logger.info("Saved %d rows x %d columns to %s", len(prices), len(prices.columns), filename)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _add_file_logging(log_dir='data'):
    """Add a file handler so logs survive terminal close during long runs."""
    from datetime import datetime as dt
    os.makedirs(log_dir, exist_ok=True)
    ts = dt.now().strftime('%Y%m%dT%H%M%S')
    log_path = os.path.join(log_dir, f'download_{ts}.log')
    handler = logging.FileHandler(log_path)
    handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s'))
    logging.getLogger().addHandler(handler)
    return log_path


def main():
    from src.logging_config import setup_logging
    setup_logging()
    parser = argparse.ArgumentParser(
        description='Build a security universe and download price data.')

    # Universe source arguments
    parser.add_argument('--asset-types', nargs='+',
                        default=['etfs'],
                        choices=['equities', 'etfs', 'funds'],
                        help='Asset types to include.')
    parser.add_argument('--countries', nargs='+', default=None,
                        help='Countries to filter equities by. '
                             'Defaults to config.INCLUDED_COUNTRIES (27 countries).')
    parser.add_argument('--all-countries', action='store_true',
                        help='Download equities from all countries '
                             '(override the default country filter).')
    parser.add_argument('--market-caps', nargs='+', default=None,
                        help='Market cap categories to include for equities. '
                             'Defaults to config.INCLUDED_MARKET_CAPS (Small Cap+).')
    parser.add_argument('--all-market-caps', action='store_true',
                        help='Download equities of all market cap sizes '
                             '(override the default market cap filter).')
    parser.add_argument('--sectors', nargs='+', default=None,
                        help='Sectors to filter equities by.')
    parser.add_argument('--exchanges', nargs='+', default=None,
                        help='Exchanges to filter equities by. '
                             'Defaults to config.INCLUDED_EXCHANGES (major US).')
    parser.add_argument('--all-exchanges', action='store_true',
                        help='Download equities from all exchanges '
                             '(override the default US exchange filter).')
    parser.add_argument('--from-csv', default=None,
                        help='Load tickers from a local CSV instead of '
                             'FinanceDatabase (e.g. data/ETFs.csv).')
    parser.add_argument('--ticker-column', default='Tickers',
                        help='Column name for tickers in the CSV.')
    parser.add_argument('--asset-type', default='etf',
                        help='Asset type label when loading from CSV '
                             '(ignored when using FinanceDatabase).')
    parser.add_argument('--exchange', default='US',
                        help='Database exchange code (US, NZX, ASX).')
    parser.add_argument('--start', default='2014-04-30',
                        help='Start date for price data.')
    parser.add_argument('--end',
                        default=__import__('datetime').date.today().isoformat(),
                        help='End date for price data (default: today).')
    parser.add_argument('--output', default='data/Prices.csv',
                        help='Output CSV file path for prices.')
    parser.add_argument('--universe-output', default='data/Securities.csv',
                        help='Output CSV for the security universe list.')
    parser.add_argument('--null-threshold', type=float, default=0.9,
                        help='Fraction of non-null values required to keep '
                             'a ticker (0-1).')
    parser.add_argument('--incremental', action='store_true',
                        help='Only download data after the latest date already '
                             'in the database.')

    # Pipeline arguments
    parser.add_argument('--subset', type=int, default=None,
                        help='Download only the first N tickers (for testing).')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be downloaded and check disk, '
                             'then exit without downloading.')
    parser.add_argument('--stage-only', action='store_true',
                        help='Download into staging DB but do not promote '
                             'to production.')
    parser.add_argument('--promote', default=None, metavar='STAGING_DB',
                        help='Promote a previously staged DB to production.')
    parser.add_argument('--rollback', default=None, metavar='BACKUP_PATH',
                        help='Restore production DB from a backup file.')
    parser.add_argument('--resume', default=None, metavar='CHECKPOINT',
                        help='Resume a previously interrupted download from '
                             'a checkpoint JSON file.')
    parser.add_argument('--rate-limit', type=float, default=None,
                        help='Seconds to wait between batches '
                             '(default: from config).')
    parser.add_argument('--no-backup', action='store_true',
                        help='Skip production DB backup before promotion.')
    parser.add_argument('--keep-staging', action='store_true',
                        help='Keep staging DB after promotion (for forensics).')
    parser.add_argument('--validate-only', action='store_true',
                        help='Run data quality checks on existing DB without '
                             'downloading.')
    parser.add_argument('--skip-validation', action='store_true',
                        help='Skip data quality checks after download.')
    args = parser.parse_args()

    from src import db, config

    # ── Apply default filters for equities ───────────────────────────────
    if args.all_countries:
        args.countries = None
        logger.info("--all-countries: downloading equities from all countries")
    elif args.countries is None and 'equities' in args.asset_types:
        args.countries = config.INCLUDED_COUNTRIES
        logger.info("Defaulting to %d configured countries (use --all-countries to override)",
                     len(args.countries))

    if args.all_market_caps:
        args.market_caps = None
        logger.info("--all-market-caps: downloading equities of all market cap sizes")
    elif args.market_caps is None and 'equities' in args.asset_types:
        args.market_caps = config.INCLUDED_MARKET_CAPS
        logger.info("Defaulting to market caps: %s (use --all-market-caps to override)",
                     ', '.join(args.market_caps))

    if args.all_exchanges:
        args.exchanges = None
        logger.info("--all-exchanges: downloading equities from all exchanges")
    elif args.exchanges is None and 'equities' in args.asset_types:
        args.exchanges = config.INCLUDED_EXCHANGES
        logger.info("Defaulting to US exchanges: %s (use --all-exchanges to override)",
                     ', '.join(args.exchanges))

    # ── Standalone operations (no download needed) ─────────────────────────

    if args.rollback:
        from src.pipeline import rollback
        rollback(args.rollback, db.DB_PATH)
        logger.info("Rollback complete.")
        return

    if args.promote:
        from src.pipeline import promote_staging, backup_database
        conn_prod = db.get_connection()
        if not args.no_backup and os.path.exists(db.DB_PATH):
            from datetime import datetime as dt
            ts = dt.now().strftime('%Y%m%d_%H%M%S')
            backup_path = f"{db.DB_PATH}.backup_{ts}"
            backup_database(conn_prod, backup_path)
        promoted = promote_staging(args.promote, conn_prod, exchange=args.exchange)
        conn_prod.close()
        logger.info("Promoted %d tickers from %s", promoted, args.promote)
        return

    if args.validate_only:
        from src.data_quality import validate_universe
        conn = db.get_connection()
        summary = validate_universe(conn, exchange=args.exchange)
        conn.close()
        logger.info("Validation summary: %s", summary)
        return

    # ── File logging (survives terminal close) ──────────────────────────
    log_path = _add_file_logging()
    logger.info("Logging to %s", log_path)

    # ── Build ticker list ──────────────────────────────────────────────────

    if args.from_csv:
        logger.info("Loading tickers from %s", args.from_csv)
        tickers_df = load_tickers(args.from_csv, args.ticker_column)
    else:
        logger.info("Building security universe from FinanceDatabase...")
        tickers_df = build_security_universe(
            asset_types=args.asset_types,
            countries=args.countries,
            sectors=args.sectors,
            exchanges=args.exchanges,
            market_caps=args.market_caps,
        )
        tickers_df.to_csv(args.universe_output, index=False)
        logger.info("Saved %d securities to %s", len(tickers_df), args.universe_output)
        for at in tickers_df['AssetType'].unique():
            count = (tickers_df['AssetType'] == at).sum()
            logger.info("  %s: %d", at, count)
        # Log country breakdown for equities
        if 'Country' in tickers_df.columns:
            equities_mask = tickers_df['AssetType'] == 'equity'
            if equities_mask.any():
                country_counts = (tickers_df.loc[equities_mask, 'Country']
                                  .value_counts()
                                  .sort_values(ascending=False))
                logger.info("Equity country breakdown (%d countries):",
                            len(country_counts))
                for country, cnt in country_counts.items():
                    logger.info("    %s: %d", country, cnt)

    # ── Determine start date ───────────────────────────────────────────────

    start = args.start
    if args.incremental:
        conn = db.get_connection()
        latest = db.get_latest_prices_date(conn, exchange=args.exchange)
        conn.close()
        if latest:
            from datetime import datetime as dt_cls, timedelta
            next_day = (dt_cls.strptime(latest, '%Y-%m-%d')
                        + timedelta(days=1)).strftime('%Y-%m-%d')
            logger.info("Incremental mode: latest DB date is %s, downloading from %s",
                         latest, next_day)
            start = next_day
        else:
            logger.info("Incremental mode: no existing data, full download from %s", start)

    # ── Dry run ────────────────────────────────────────────────────────────

    if args.dry_run:
        from src.pipeline import preflight_check
        num_tickers = len(tickers_df)
        if args.subset:
            num_tickers = min(num_tickers, args.subset)

        from datetime import datetime as dt_cls
        start_dt = dt_cls.strptime(start, '%Y-%m-%d')
        end_dt = dt_cls.strptime(args.end, '%Y-%m-%d')
        est_days = int((end_dt - start_dt).days * 252 / 365)

        data_dir = os.path.dirname(db.DB_PATH)
        staging_path = os.path.join(data_dir, 'staging_dryrun.db')
        preflight = preflight_check(db.DB_PATH, staging_path,
                                    num_tickers, est_days)

        logger.info("DRY RUN — would download %d tickers, %s to %s",
                     num_tickers, start, args.end)
        logger.info("  Estimated staging DB: %.2f GB", preflight['estimated_staging_gb'])
        logger.info("  Production DB: %.2f GB", preflight['prod_size_gb'])
        logger.info("  Total space needed: %.2f GB", preflight['estimated_total_gb'])
        logger.info("  Disk available: %.2f GB", preflight['available_gb'])
        if preflight['ok']:
            logger.info("  Preflight: PASS")
        else:
            for w in preflight['warnings']:
                logger.warning("  Preflight: FAIL — %s", w)
        return

    # ── Resume from checkpoint ─────────────────────────────────────────────

    checkpoint_path = None
    staging_db_path = None
    if args.resume:
        from src.pipeline import load_checkpoint
        checkpoint = load_checkpoint(args.resume)
        if not checkpoint:
            logger.error("Checkpoint file not found or empty: %s", args.resume)
            return
        checkpoint_path = args.resume
        staging_db_path = checkpoint.get('staging_db')

    # ── Run pipeline per asset type ────────────────────────────────────────

    from src.pipeline import run_pipeline

    logger.info("Downloading prices for %d tickers...", len(tickers_df))

    if 'AssetType' in tickers_df.columns:
        all_manifests = []
        for fd_type, group in tickers_df.groupby('AssetType'):
            db_type = ASSET_TYPE_MAP.get(fd_type, fd_type)
            ticker_list = group[args.ticker_column].tolist()
            names = None
            if 'Name' in group.columns:
                names = dict(zip(group[args.ticker_column], group['Name']))
            countries = None
            if 'Country' in group.columns:
                countries = dict(zip(group[args.ticker_column], group['Country']))
            logger.info("Running pipeline for %d %s tickers...",
                        len(ticker_list), db_type)
            manifest = run_pipeline(
                ticker_list, exchange=args.exchange, asset_type=db_type,
                start=start, end=args.end, null_threshold=args.null_threshold,
                names=names, countries=countries,
                subset=args.subset,
                stage_only=args.stage_only,
                skip_validation=args.skip_validation,
                keep_staging=args.keep_staging,
                no_backup=args.no_backup,
                checkpoint_path=checkpoint_path,
                staging_db_path=staging_db_path,
                rate_limit_delay=args.rate_limit,
            )
            all_manifests.append((db_type, manifest))

        for db_type, manifest in all_manifests:
            dl = manifest.get('download_result', {})
            logger.info("  %s: status=%s, saved=%s, failed_batches=%s",
                         db_type, manifest['status'],
                         dl.get('saved_tickers', 'N/A'),
                         len(dl.get('failed_batches', [])))
    else:
        ticker_list = tickers_df[args.ticker_column].tolist()
        names = None
        if 'Name' in tickers_df.columns:
            names = dict(zip(tickers_df[args.ticker_column], tickers_df['Name']))
        manifest = run_pipeline(
            ticker_list, exchange=args.exchange, asset_type=args.asset_type,
            start=start, end=args.end, null_threshold=args.null_threshold,
            names=names,
            subset=args.subset,
            stage_only=args.stage_only,
            skip_validation=args.skip_validation,
            keep_staging=args.keep_staging,
            no_backup=args.no_backup,
            checkpoint_path=checkpoint_path,
            staging_db_path=staging_db_path,
            rate_limit_delay=args.rate_limit,
        )
        dl = manifest.get('download_result', {})
        logger.info("Pipeline: status=%s, saved=%s, failed_batches=%s",
                     manifest['status'],
                     dl.get('saved_tickers', 'N/A'),
                     len(dl.get('failed_batches', [])))

    # Step 4: Export to CSV for backward compatibility (only after promotion)
    if not args.stage_only:
        conn = db.get_connection()
        prices_df = db.load_prices(conn, exchange=args.exchange)
        conn.close()
        if not prices_df.empty:
            save_to_csv(prices_df, args.output)
        else:
            logger.warning("No prices in database to export.")

    # ── Final summary ─────────────────────────────────────────────────────
    _log_final_summary(all_manifests if 'AssetType' in tickers_df.columns
                       else [('all', manifest)], log_path)


def _log_final_summary(manifests, log_path):
    """Print a clear final summary with key stats and file locations."""
    logger.info("=" * 60)
    logger.info("DOWNLOAD COMPLETE")
    logger.info("=" * 60)

    total_saved = 0
    total_failed_batches = 0
    total_failed_tickers = 0
    for db_type, m in manifests:
        dl = m.get('download_result', {})
        saved = dl.get('saved_tickers', 0)
        failed = dl.get('failed_batches', [])
        failed_ticker_count = sum(len(fb['tickers']) for fb in failed)
        duration = m.get('duration_seconds', 0)
        total_saved += saved
        total_failed_batches += len(failed)
        total_failed_tickers += failed_ticker_count

        minutes, secs = divmod(int(duration), 60)
        hours, minutes = divmod(minutes, 60)

        logger.info("  %s: status=%s, saved=%d, failed=%d tickers (%d batches), "
                     "duration=%dh%02dm%02ds",
                     db_type, m['status'], saved,
                     failed_ticker_count, len(failed),
                     hours, minutes, secs)

        # Log validation summary if present
        validation = m.get('validation')
        if validation:
            logger.info("    validation: %d active, %d excluded out of %d total",
                        validation.get('total_active', 0),
                        validation.get('total_excluded', 0),
                        validation.get('total_tickers', 0))

        # Log manifest path
        run_id = m.get('run_id', 'unknown')
        logger.info("    manifest: data/manifest_%s.json", run_id)

        # Log failed batch details
        if failed:
            logger.info("    failed batches:")
            for fb in failed:
                tickers_preview = ', '.join(fb['tickers'][:5])
                suffix = (f' ... +{len(fb["tickers"]) - 5} more'
                          if len(fb['tickers']) > 5 else '')
                logger.info("      batch %d (%d tickers): %s%s",
                            fb['batch_num'], len(fb['tickers']),
                            tickers_preview, suffix)

    logger.info("-" * 60)
    logger.info("Totals: %d saved, %d failed tickers", total_saved, total_failed_tickers)
    logger.info("Log file: %s", log_path)
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
