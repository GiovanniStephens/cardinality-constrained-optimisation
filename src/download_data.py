"""
Builds a security universe from FinanceDatabase and downloads price data
from Yahoo Finance.

Supports equities, ETFs, and funds. Can also load tickers from a local CSV
for working with previously scraped lists.
"""

import argparse
import logging

import financedatabase as fd
import pandas as pd
import yfinance as yf

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
                 exchanges=None) -> pd.DataFrame:
    """
    Retrieves equity tickers from FinanceDatabase.

    :param countries: country or list of countries to filter by.
    :param sectors: sector or list of sectors to filter by.
    :param industries: industry or list of industries to filter by.
    :param exchanges: exchange or list of exchanges to filter by (post-filter).
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
    return _filter_by_exchange(equities.select(**kwargs), exchanges)


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
                            etf_categories=None,
                            etf_category_groups=None) -> pd.DataFrame:
    """
    Builds a combined universe of securities from multiple asset types.

    :param asset_types: list of asset types to include.
        Options: 'equities', 'etfs', 'funds'. Defaults to all three.
    :param countries: country filter (applies to equities).
    :param sectors: sector filter (applies to equities).
    :param industries: industry filter (applies to equities).
    :param exchanges: exchange filter (applies to all asset types).
    :param etf_categories: category filter (applies to ETFs).
    :param etf_category_groups: category group filter (applies to ETFs).
    :returns: DataFrame with columns ['Tickers', 'Name', 'Country', 'AssetType'].
    """
    if asset_types is None:
        asset_types = ['equities', 'etfs', 'funds']

    all_securities = []

    if 'equities' in asset_types:
        equities = get_equities(countries=countries, sectors=sectors,
                                industries=industries, exchanges=exchanges)
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
                        categories=etf_categories, exchanges=exchanges)
        if not etfs.empty:
            etf_df = pd.DataFrame({
                'Tickers': etfs.index,
                'Name': etfs['name'] if 'name' in etfs.columns else '',
                'Country': '',
                'AssetType': 'etf'
            })
            all_securities.append(etf_df)

    if 'funds' in asset_types:
        funds = get_funds(exchanges=exchanges)
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
    batch_prices = {}
    for ticker in tickers:
        try:
            if len(tickers) == 1:
                batch_prices[ticker] = prices["Close"].tolist()
            else:
                batch_prices[ticker] = prices[ticker]["Close"].tolist()
        except (KeyError, TypeError):
            logger.warning("No data returned for ticker '%s'; skipping.", ticker)
    if not batch_prices:
        return None
    return pd.DataFrame(batch_prices)


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
        try:
            batch_df = _download_batch(batch, start, end)
        except Exception as e:
            raise ConnectionError(
                f"Failed to download data from Yahoo Finance: {e}") from e
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
    :returns: dict with keys total_tickers, saved_tickers, failed_batches.
    """
    from src import db

    batches = [
        tickers[i : i + batch_size]
        for i in range(0, len(tickers), batch_size)
    ]
    total_saved = 0
    failed_batches = []

    for batch_num, batch in enumerate(batches, 1):
        logger.info("Batch %d/%d (%d tickers)...", batch_num, len(batches), len(batch))
        batch_df = None
        for attempt in range(1, max_retries + 1):
            try:
                batch_df = _download_batch(batch, start, end)
                break
            except Exception as e:
                logger.warning("Batch %d attempt %d/%d failed: %s",
                               batch_num, attempt, max_retries, e)
                if attempt < max_retries:
                    import time
                    time.sleep(2 ** attempt)

        if batch_df is None or batch_df.empty:
            logger.warning("Batch %d: no data returned, skipping.", batch_num)
            failed_batches.append(batch_num)
            continue

        # Filter out tickers with too many nulls
        threshold = int(len(batch_df) * null_threshold)
        batch_df = batch_df.dropna(axis=1, thresh=threshold)
        if batch_df.empty:
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
        total_saved += batch_df.shape[1]
        logger.info("Batch %d: saved %d tickers (total: %d)",
                     batch_num, batch_df.shape[1], total_saved)

    return {
        'total_tickers': len(tickers),
        'saved_tickers': total_saved,
        'failed_batches': failed_batches,
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

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description='Build a security universe and download price data.')
    parser.add_argument('--asset-types', nargs='+',
                        default=['etfs'],
                        choices=['equities', 'etfs', 'funds'],
                        help='Asset types to include.')
    parser.add_argument('--countries', nargs='+', default=None,
                        help='Countries to filter equities by.')
    parser.add_argument('--sectors', nargs='+', default=None,
                        help='Sectors to filter equities by.')
    parser.add_argument('--exchanges', nargs='+', default=None,
                        help='Exchanges to filter by.')
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
    parser.add_argument('--end', default='2025-04-30',
                        help='End date for price data.')
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
    args = parser.parse_args()

    from src import db

    # Step 1: Get tickers
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
        )
        tickers_df.to_csv(args.universe_output, index=False)
        logger.info("Saved %d securities to %s", len(tickers_df), args.universe_output)
        for asset_type in tickers_df['AssetType'].unique():
            count = (tickers_df['AssetType'] == asset_type).sum()
            logger.info("  %s: %d", asset_type, count)

    # Step 2: Determine start date
    conn = db.get_connection()
    start = args.start
    if args.incremental:
        latest = db.get_latest_prices_date(conn, exchange=args.exchange)
        if latest:
            # Start from the day after the latest date in the DB
            from datetime import datetime, timedelta
            next_day = (datetime.strptime(latest, '%Y-%m-%d')
                        + timedelta(days=1)).strftime('%Y-%m-%d')
            logger.info("Incremental mode: latest DB date is %s, downloading from %s",
                         latest, next_day)
            start = next_day
        else:
            logger.info("Incremental mode: no existing data, full download from %s", start)

    # Step 3: Download and save to database per asset type
    logger.info("Downloading prices for %d tickers...", len(tickers_df))

    if 'AssetType' in tickers_df.columns:
        # FinanceDatabase path: download per asset type group
        all_results = []
        for fd_type, group in tickers_df.groupby('AssetType'):
            db_type = ASSET_TYPE_MAP.get(fd_type, fd_type)
            ticker_list = group[args.ticker_column].tolist()
            names = None
            if 'Name' in group.columns:
                names = dict(zip(group[args.ticker_column], group['Name']))
            countries = None
            if 'Country' in group.columns:
                countries = dict(zip(group[args.ticker_column], group['Country']))
            logger.info("Downloading %d %s tickers...", len(ticker_list), db_type)
            result = download_and_save(
                ticker_list, conn, exchange=args.exchange, asset_type=db_type,
                start=start, end=args.end, null_threshold=args.null_threshold,
                names=names, countries=countries,
            )
            all_results.append((db_type, result))

        for db_type, result in all_results:
            logger.info("  %s: %d/%d saved, %d failed batches",
                         db_type, result['saved_tickers'],
                         result['total_tickers'], len(result['failed_batches']))
    else:
        # CSV path: single asset type
        ticker_list = tickers_df[args.ticker_column].tolist()
        names = None
        if 'Name' in tickers_df.columns:
            names = dict(zip(tickers_df[args.ticker_column], tickers_df['Name']))
        result = download_and_save(
            ticker_list, conn, exchange=args.exchange,
            asset_type=args.asset_type,
            start=start, end=args.end, null_threshold=args.null_threshold,
            names=names,
        )
        logger.info("Saved %d/%d tickers, %d failed batches",
                     result['saved_tickers'], result['total_tickers'],
                     len(result['failed_batches']))

    # Step 4: Export to CSV for backward compatibility
    prices_df = db.load_prices(conn, exchange=args.exchange)
    conn.close()
    if not prices_df.empty:
        save_to_csv(prices_df, args.output)
    else:
        logger.warning("No prices in database to export.")


if __name__ == '__main__':
    main()
