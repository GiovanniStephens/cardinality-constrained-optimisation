"""
Security universe building from FinanceDatabase and ticker filtering/loading.

Provides functions to query FinanceDatabase for equities, ETFs, funds, cryptos,
and currencies, combine them into a unified universe DataFrame, and filter out
unwanted tickers (warrants, units, SPACs, etc.).
"""

import logging
import re

import financedatabase as fd
import pandas as pd

from src.config import TICKER_EXCLUDE_SUFFIXES, TICKER_EXCLUDE_NAME_PATTERNS

logger = logging.getLogger(__name__)

# Maps FinanceDatabase asset type labels to DB-canonical names.
# DB migration uses 'stock' for equities (see db._migrate_ticker_list).
ASSET_TYPE_MAP = {
    'equity': 'stock', 'etf': 'etf', 'fund': 'fund',
    'crypto': 'crypto', 'currency': 'currency',
}


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


def get_cryptos() -> pd.DataFrame:
    """Retrieves cryptocurrency tickers from FinanceDatabase."""
    return fd.Cryptos().select()


def get_currencies() -> pd.DataFrame:
    """Retrieves currency pair tickers from FinanceDatabase."""
    return fd.Currencies().select()


ALL_ASSET_TYPES = ['equities', 'etfs', 'funds', 'cryptos', 'currencies']


def build_security_universe(asset_types=None, countries=None, sectors=None,
                            industries=None, exchanges=None,
                            market_caps=None, etf_categories=None,
                            etf_category_groups=None) -> pd.DataFrame:
    """
    Builds a combined universe of securities from multiple asset types.

    :param asset_types: list of asset types to include. Defaults to ALL_ASSET_TYPES.
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
        asset_types = ALL_ASSET_TYPES

    all_securities = []

    def _append(df, asset_type):
        if df.empty:
            return
        row = {
            'Tickers': df.index,
            'Name': df['name'] if 'name' in df.columns else '',
            'Country': df['country'] if 'country' in df.columns else '',
            'AssetType': asset_type,
        }
        # Carry through sector/industry (equities) and category_group/category (ETFs)
        for col in ('sector', 'industry', 'category_group', 'category'):
            if col in df.columns:
                row[col] = df[col]
        all_securities.append(pd.DataFrame(row))

    if 'equities' in asset_types:
        _append(get_equities(countries=countries, sectors=sectors,
                             industries=industries, exchanges=exchanges,
                             market_caps=market_caps), 'equity')

    if 'etfs' in asset_types:
        _append(get_etfs(category_groups=etf_category_groups,
                         categories=etf_categories), 'etf')

    if 'funds' in asset_types:
        _append(get_funds(), 'fund')

    if 'cryptos' in asset_types:
        _append(get_cryptos(), 'crypto')

    if 'currencies' in asset_types:
        _append(get_currencies(), 'currency')

    if not all_securities:
        return pd.DataFrame(columns=['Tickers', 'Name', 'Country', 'AssetType'])

    combined = pd.concat(all_securities, ignore_index=True)
    combined = combined.drop_duplicates(subset='Tickers')
    return combined


def filter_unwanted_tickers(df, ticker_column='Tickers', name_column='Name',
                            skip_suffix_filter=False, skip_name_filter=False):
    """Remove warrants, units, preferred shares, rights, and SPACs.

    Applies two tiers of regex filtering (zero API calls):
      1a. Ticker suffix patterns (e.g. -WT, -UN, -PA, -RT)
      1b. Name patterns (e.g. "Acquisition Corp", "Blank Check")

    :param df: DataFrame with at least a ticker column.
    :param ticker_column: column containing ticker symbols.
    :param name_column: column containing security names.
    :param skip_suffix_filter: if True, skip suffix regex filtering.
    :param skip_name_filter: if True, skip name pattern filtering.
    :returns: (filtered_df, removed_df) — both DataFrames.
    """
    if df.empty:
        return df.copy(), df.iloc[:0].copy()

    remove_mask = pd.Series(False, index=df.index)

    # Tier 1a: suffix regex on ticker symbols
    if not skip_suffix_filter and ticker_column in df.columns:
        suffix_re = re.compile(TICKER_EXCLUDE_SUFFIXES)
        remove_mask |= df[ticker_column].str.contains(suffix_re, na=False)

    # Tier 1b: name patterns for SPACs/shell companies
    if not skip_name_filter and name_column in df.columns:
        combined_pattern = '|'.join(TICKER_EXCLUDE_NAME_PATTERNS)
        name_re = re.compile(combined_pattern, re.IGNORECASE)
        remove_mask |= df[name_column].str.contains(name_re, na=False)

    removed = df[remove_mask].copy()
    filtered = df[~remove_mask].copy()
    return filtered, removed


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
