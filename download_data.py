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


# ---------------------------------------------------------------------------
# Universe building (from FinanceDatabase)
# ---------------------------------------------------------------------------

def get_equities(countries=None, sectors=None, industries=None,
                 exchanges=None) -> pd.DataFrame:
    """
    Retrieves equity tickers from FinanceDatabase.

    :param countries: country or list of countries to filter by.
    :param sectors: sector or list of sectors to filter by.
    :param industries: industry or list of industries to filter by.
    :param exchanges: exchange or list of exchanges to filter by.
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
    if exchanges:
        kwargs['exchange'] = exchanges
    return equities.select(**kwargs)


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
    if exchanges:
        kwargs['exchange'] = exchanges
    return etfs.select(**kwargs)


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
    if exchanges:
        kwargs['exchange'] = exchanges
    return funds.select(**kwargs)


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
    :returns: DataFrame with columns ['Tickers', 'Name', 'AssetType'].
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
                'AssetType': 'etf'
            })
            all_securities.append(etf_df)

    if 'funds' in asset_types:
        funds = get_funds(exchanges=exchanges)
        if not funds.empty:
            fund_df = pd.DataFrame({
                'Tickers': funds.index,
                'Name': funds['name'] if 'name' in funds.columns else '',
                'AssetType': 'fund'
            })
            all_securities.append(fund_df)

    if not all_securities:
        return pd.DataFrame(columns=['Tickers', 'Name', 'AssetType'])

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
        logger.info("Downloading batch %d/%d (%d tickers)...", batch_num, len(batches), len(batch))
        tickers_str = " ".join(batch)
        try:
            prices = yf.download(
                tickers_str,
                interval="1d",
                group_by="ticker",
                start=start,
                end=end,
            )
        except Exception as e:
            raise ConnectionError(f"Failed to download data from Yahoo Finance: {e}") from e
        for ticker in batch:
            try:
                if len(batch) == 1:
                    all_prices[ticker] = prices["Close"].tolist()
                else:
                    all_prices[ticker] = prices[ticker]["Close"].tolist()
            except (KeyError, TypeError):
                logger.warning("No data returned for ticker '%s'; skipping.", ticker)

    if not all_prices:
        raise ValueError("No valid price data could be extracted for any ticker.")
    prices_df = pd.DataFrame(all_prices)
    return prices_df


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
                             'FinanceDatabase (e.g. Data/ETFs.csv).')
    parser.add_argument('--ticker-column', default='Tickers',
                        help='Column name for tickers in the CSV.')
    parser.add_argument('--start', default='2014-04-30',
                        help='Start date for price data.')
    parser.add_argument('--end', default='2025-04-30',
                        help='End date for price data.')
    parser.add_argument('--output', default='Data/ETF_Prices.csv',
                        help='Output CSV file path for prices.')
    parser.add_argument('--universe-output', default='Data/Securities.csv',
                        help='Output CSV for the security universe list.')
    parser.add_argument('--null-threshold', type=float, default=0.9,
                        help='Fraction of non-null values required to keep '
                             'a ticker (0-1).')
    args = parser.parse_args()

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

    # Step 2: Download prices
    logger.info("Downloading prices for %d tickers...", len(tickers_df))
    try:
        prices = download_data(tickers_df, ticker_column=args.ticker_column,
                               start=args.start, end=args.end)
    except Exception:
        logger.exception("Failed to download price data from Yahoo Finance")
        return
    threshold = int(len(prices) * args.null_threshold)
    filtered_prices = prices.dropna(axis=1, thresh=threshold)
    save_to_csv(filtered_prices, args.output)
    logger.info("Saved %d tickers (%d dropped) to %s",
                filtered_prices.shape[1],
                prices.shape[1] - filtered_prices.shape[1],
                args.output)


if __name__ == '__main__':
    main()
