import logging

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def load_tickers(filename: str, ticker_column: str = 'Tickers') -> pd.DataFrame:
    """
    Loads the list of tickers from a local CSV file.

    :param filename: path to CSV file containing tickers.
    :param ticker_column: name of the column containing ticker symbols.
    :returns: list of tickers as a pandas DataFrame.
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


# Keep backward compatibility
load_etfs = load_tickers


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    tickers = load_tickers('Data/ETFs_Full.csv', ticker_column='Tickers')
    logger.info("Loaded %d tickers from list", len(tickers))
    try:
        prices = download_data(tickers, ticker_column='Tickers')
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
