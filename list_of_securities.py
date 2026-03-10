"""
Builds a universe of investable securities using the FinanceDatabase package.

Supports equities, ETFs, and funds across multiple markets and regions.
Replaces the ETF-only list_of_ETFs.py with a broader security universe.
"""

import argparse

import financedatabase as fd
import pandas as pd


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
    result = equities.select(**kwargs)
    return result


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
    result = etfs.select(**kwargs)
    return result


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
    result = funds.select(**kwargs)
    return result


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


def main():
    parser = argparse.ArgumentParser(
        description='Build a security universe from FinanceDatabase.')
    parser.add_argument('--asset-types', nargs='+',
                        default=['equities', 'etfs', 'funds'],
                        choices=['equities', 'etfs', 'funds'],
                        help='Asset types to include.')
    parser.add_argument('--countries', nargs='+', default=None,
                        help='Countries to filter equities by.')
    parser.add_argument('--sectors', nargs='+', default=None,
                        help='Sectors to filter equities by.')
    parser.add_argument('--exchanges', nargs='+', default=None,
                        help='Exchanges to filter by.')
    parser.add_argument('--output', default='Data/Securities.csv',
                        help='Output CSV file path.')
    args = parser.parse_args()

    securities = build_security_universe(
        asset_types=args.asset_types,
        countries=args.countries,
        sectors=args.sectors,
        exchanges=args.exchanges,
    )

    securities.to_csv(args.output, index=False)
    print(f"Saved {len(securities)} securities to {args.output}")

    for asset_type in securities['AssetType'].unique():
        count = (securities['AssetType'] == asset_type).sum()
        print(f"  {asset_type}: {count}")


if __name__ == '__main__':
    main()
