import pandas as pd
import yfinance as yf


def load_etfs() -> pd.DataFrame:
    """
    Loads the list of ETFs from my local file.

    :returns: list of ETFs as a pandas DataFrame 
    """
    etfs = pd.read_csv('ETFs.csv')
    return etfs


if __name__ == '__main__':
    etfs = load_etfs()
    for etf in etfs.values:
        print(etf)