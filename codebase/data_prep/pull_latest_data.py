
import os, contextlib, sys
import pandas as pd
import yfinance as yf
from pathlib import Path

sys.path.insert(0, '../')

from codebase import constants

DATASET_FOLDER = constants.DATASET_FOLDER

class StockScraper:

    def __init__(self, dataset_folder=DATASET_FOLDER) -> None:
        self.dataset_folder = dataset_folder
        self.url = 'https://en.wikipedia.org/wiki/NASDAQ-100#Components'
        self.col_list = ['Date', 'Close', 'Volume']

        # create a folder for data storage
        if not os.path.exists(self.dataset_folder):
            os.mkdir(self.dataset_folder)

    def scrape(self):
        html = pd.read_html(self.url, header=0)
        series = html[4]["Ticker"]
        symbols = series.to_list()
            
        # download prices history
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull): 
                for i, symbol in enumerate(symbols):
                    data = yf.download(symbol, period='max')
                    data.to_csv(f'{self.dataset_folder}/{symbol}.csv')

if __name__ == '__main__':
    scraper = StockScraper(dataset_folder=DATASET_FOLDER)
    scraper.scrape()

