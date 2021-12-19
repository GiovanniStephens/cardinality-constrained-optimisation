import time
import pandas as pd
from bs4 import BeautifulSoup as bs
from selenium import webdriver
from tqdm import tqdm


BASE_URL = 'https://etfdb.com/screener/'
# Only ETFs with at least 3 years of data are included.
PARAMS = {'three_ytd_start': '-99'}


def set_up_driver():
    """
    Sets up the Chrome webdriver.
    You can find the lastest version here:
    https://chromedriver.chromium.org/downloads

    :return: webdriver object.
    """
    options = webdriver.ChromeOptions()
    options.add_argument('headless')
    options.add_argument("disable-gpu")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-infobars")

    driver = webdriver.Chrome(chrome_options=options)
    return driver


def build_url(page_num: int) -> str:
    """
    Builds the URL for the page number.

    :page_num: int of the page number.
    :return: string of the URL.
    """
    url = BASE_URL + '#page='\
                   + str(page_num) \
                   + '&' \
                   + '&'.join(PARAMS.keys()) \
                   + '=' \
                   + '&'.join(PARAMS.values())
    return url


def get_num_pages(driver: webdriver) -> int:
    """
    Gets the number of pages in the ETF database.

    :driver: webdriver object.
    :return: int of the number of pages.
    """
    url = build_url(1)
    driver.get(url)
    html = driver.page_source
    soup = bs(html, 'html.parser')
    num_etfs = soup.find_all('span',
                             attrs={'data-screener-filters-total':
                                    True})[0].text
    num_pages = int(num_etfs) // 25 + 1
    return num_pages


def get_all_ETFs():
    """
    Scrapes all the ETFs from ETFdb.com.

    :return: pandas dataframe of the ETFs.
    """
    driver = set_up_driver()
    num_pages = get_num_pages(driver)
    etf_df = pd.DataFrame()
    for i in tqdm(range(num_pages)):
        url = build_url(i+1)
        etf_df = etf_df.append(get_ETFs(driver, url))
    return etf_df


def get_page_soup(driver, url):
    """
    Uses Selenium to render the page, and
    download the page content.

    :driver: webdriver object.
    :url: string of the URL.
    :return: BeautifulSoup object.
    """
    driver.get(url)
    # For some reason the page loading was not refreshing the table.
    # I tell Selenium to load the page twice.
    # This is a hacky fix.
    driver.get(url)
    # Wait for the page table to update
    time.sleep(0.5)
    html = driver.page_source
    soup = bs(html, 'html.parser')
    return soup


def get_ETFs(driver, url):
    """
    Gets the ETFs from the page.

    :driver: webdriver object.
    :url: string of the URL.
    :return: pandas dataframe of the ETFs.
    """
    soup = get_page_soup(driver, url)
    etfs = soup.find_all('td',
                         attrs={'data-th': 'Symbol'})
    etf_names = soup.find_all('td',
                              attrs={'data-th': 'ETF Name'})
    asset_classes = soup.find_all('td',
                                  attrs={'data-th': 'Asset Class'})
    etf_list = []
    for i in range(len(etfs)):
        etf_list.append({'ETF': etfs[i].text,
                         'Name': etf_names[i].text,
                         'Asset Class': asset_classes[i].text})
    etf_df = pd.DataFrame(etf_list)
    return etf_df


def main():
    """
    Main function.
    """
    etf_df = get_all_ETFs()
    etf_df.to_csv('etfs.csv', index=False)
    etf_df = get_all_ETFs()
    etf_df.to_csv('ETFs.csv')


if __name__ == '__main__':
    main()
