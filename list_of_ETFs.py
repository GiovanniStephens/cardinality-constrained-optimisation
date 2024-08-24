from etfpy import get_available_etfs_list
import pandas as pd


def main():
    """
    Main function.
    """
    etfs_list = get_available_etfs_list()
    etf_df = pd.DataFrame(etfs_list, columns=['Ticker'])
    etf_df.to_csv('Data/ETFs_.csv', index=False)


if __name__ == '__main__':
    main()
