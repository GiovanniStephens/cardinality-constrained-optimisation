import logging

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def load_etfs(filename: str) -> pd.DataFrame:
    """
    Loads the list of ETFs from my local file.

    :returns: list of ETFs as a pandas DataFrame.
    """
    etfs = pd.read_csv(filename)
    if etfs.empty:
        raise ValueError(f"ETF list file '{filename}' is empty.")
    if 'Tickers' not in etfs.columns:
        raise KeyError("ETF file must contain a 'Tickers' column.")
    return etfs


def download_data(
    etfs: pd.DataFrame,
    ticker_col: str = "Tickers",
    start: str = "2014-04-30",
    end: str = "2025-04-30",
    batch_size: int = 500,
) -> pd.DataFrame:
    """
    Downloads data from Yahoo Finance for the given list of tickers.

    Processes tickers in batches to avoid timeouts with large ticker lists.

    :etfs: DataFrame containing ticker symbols.
    :ticker_col: name of the column containing ticker symbols.
    :start: start date for the download (YYYY-MM-DD).
    :end: end date for the download (YYYY-MM-DD).
    :batch_size: number of tickers to download per batch.
    :returns: daily price data for the given list of tickers.
    """
    all_tickers = etfs[ticker_col].tolist()
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

    :prices: daily price data for the given list of ETFs.
    :filename: name of the file to save the data to.
    """
    prices.to_csv(filename)
    logger.info("Saved %d rows x %d columns to %s", len(prices), len(prices.columns), filename)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    etfs = load_etfs('Data/ETFs_Full.csv')
    logger.info("Loaded %d ETFs from list", len(etfs))
    try:
        prices = download_data(etfs)
    except Exception:
        logger.exception("Failed to download price data from Yahoo Finance")
        return
    logger.info("Downloaded price data: %d rows x %d columns", *prices.shape)
    filtered_prices = prices.dropna(axis=1, thresh=90)
    dropped = prices.shape[1] - filtered_prices.shape[1]
    if dropped:
        logger.info("Dropped %d columns with insufficient data", dropped)
    save_to_csv(filtered_prices, 'Data/ETF_Prices.csv')
    logger.info("Saved filtered prices to Data/ETF_Prices.csv")

    import db
    conn = db.get_connection()
    ds_id = db.save_prices(conn, filtered_prices, exchange='US', asset_type='etf')
    print(f"Prices saved to database (data_source id={ds_id})")
    conn.close()


if __name__ == '__main__':
    main()
