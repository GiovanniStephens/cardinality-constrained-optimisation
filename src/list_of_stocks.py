import io
import json
import re
import urllib.request

import pandas as pd


def fetch_nasdaq_traded() -> pd.DataFrame:
    """
    Downloads the NASDAQ traded symbols file from nasdaqtrader.com.

    Returns a DataFrame with columns including Symbol, Security Name,
    Listing Exchange, ETF flag, Test Issue flag, etc.
    """
    url = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt"
    response = urllib.request.urlopen(url)
    raw = response.read().decode("utf-8")
    df = pd.read_csv(
        io.StringIO(raw),
        sep="|",
        skipfooter=1,
        engine="python",
    )
    return df


def filter_common_stocks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters a NASDAQ traded DataFrame to common stocks only.

    Applies the following filters:
    - Nasdaq Traded == 'Y' and ETF == 'N' and Test Issue == 'N'
    - Symbol matches ^[A-Z]{1,5}$ (excludes preferred shares, warrants, units, rights)
    - Security Name excludes keywords: Warrant, Rights, Units, Preferred, Debenture, Notes
    """
    mask = (
        (df["Nasdaq Traded"] == "Y")
        & (df["ETF"] == "N")
        & (df["Test Issue"] == "N")
    )
    filtered = df[mask].copy()

    symbol_pattern = re.compile(r"^[A-Z]{1,5}$")
    filtered = filtered[filtered["Symbol"].apply(lambda s: bool(symbol_pattern.match(str(s))))]

    exclude_keywords = ["Warrant", "Rights", "Units", "Preferred", "Debenture", "Notes"]
    pattern = "|".join(exclude_keywords)
    filtered = filtered[~filtered["Security Name"].str.contains(pattern, case=False, na=False)]

    return filtered


def fetch_sec_tickers() -> dict:
    """
    Fetches the SEC EDGAR company tickers JSON.

    Returns a dict mapping CIK numbers to {cik_str, ticker, title}.
    Useful for cross-referencing NASDAQ data with SEC filings.
    """
    url = "https://www.sec.gov/files/company_tickers.json"
    req = urllib.request.Request(url, headers={"User-Agent": "MyApp/1.0"})
    response = urllib.request.urlopen(req)
    data = json.loads(response.read().decode("utf-8"))
    return data


def fetch_sec_tickers_with_exchange() -> dict:
    """
    Fetches the SEC EDGAR company tickers with exchange information.

    Returns a dict with ticker, name, CIK, and exchange fields.
    Useful for filtering by specific exchanges (NYSE, NASDAQ, etc.).
    """
    url = "https://www.sec.gov/files/company_tickers_exchange.json"
    req = urllib.request.Request(url, headers={"User-Agent": "MyApp/1.0"})
    response = urllib.request.urlopen(req)
    data = json.loads(response.read().decode("utf-8"))
    return data


def main():
    print("Fetching NASDAQ traded symbols...")
    df = fetch_nasdaq_traded()
    print(f"Total symbols in NASDAQ file: {len(df)}")

    stocks = filter_common_stocks(df)
    print(f"Common stocks after filtering: {len(stocks)}")

    tickers = stocks["Symbol"].sort_values().reset_index(drop=True)
    out = pd.DataFrame(tickers.values, columns=["Tickers"])
    out.to_csv("Data/US_Stocks.csv", index=False)
    print(f"Saved {len(out)} tickers to Data/US_Stocks.csv")

    print(f"\nFirst 10: {tickers.head(10).tolist()}")
    print(f"Last 10:  {tickers.tail(10).tolist()}")


if __name__ == "__main__":
    main()
