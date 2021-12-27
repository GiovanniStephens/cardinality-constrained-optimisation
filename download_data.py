import pandas as pd
import numpy as np
import yfinance as yf


def load_etfs(filename: str) -> pd.DataFrame:
    """
    Loads the list of ETFs from my local file.

    :returns: list of ETFs as a pandas DataFrame 
    """
    etfs = pd.read_csv(filename)
    return etfs


def download_data(etfs: pd.DataFrame) -> pd.DataFrame:
    """
    Downloads data from Yahoo Finance for the given list of ETFs.

    :etfs: list of ETFs as a pandas DataFrame
    :returns: daily price data for the given list of ETFs.
    """
    tickers = ' '.join(etfs['Tickers'].to_list())
    prices = yf.download(tickers, period='3y', interval='1d', group_by='ticker')
    adj_closing_prices = []
    for ticker in etfs['Tickers']:
        adj_closing_prices.append(prices[ticker]['Adj Close'].to_list())
    prices_df = pd.DataFrame(np.transpose(adj_closing_prices), columns=etfs['Tickers'])
    return prices_df


def save_to_csv(prices: pd.DataFrame, filename: str) -> None:
    """
    Saves the given DataFrame to a CSV file.

    :prices: daily price data for the given list of ETFs.
    :filename: name of the file to save the data to.
    """
    prices.to_csv(filename)


def main():
    etfs = load_etfs('ETFs.csv')
    prices = download_data(etfs)
    save_to_csv(prices, 'ETF_Prices.csv')


if __name__ == '__main__':
    main()