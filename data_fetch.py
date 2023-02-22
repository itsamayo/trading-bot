import os
import requests

from dotenv import load_dotenv

load_dotenv()

# set up your env variables
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
STOCK_SYMBOL = os.getenv('STOCK_SYMBOL')

# define the API URL and parameters
URL = 'https://www.alphavantage.co/query'
PARAMS = {
    'function': 'TIME_SERIES_DAILY_ADJUSTED',
    'symbol': STOCK_SYMBOL,
    'outputsize': 'full',
    'datatype': 'csv',
    'apikey': ALPHA_VANTAGE_API_KEY
}

def fetch_historical_data():
    # send a GET request to the API
    response = requests.get(URL, params=PARAMS)

    # return the historical data
    return response.content