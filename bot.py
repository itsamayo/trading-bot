import pandas as pd
import numpy as np
import requests
import io
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# define the API URL and parameters
url = 'https://www.alphavantage.co/query'
params = {
    'function': 'TIME_SERIES_DAILY_ADJUSTED',
    'symbol': os.getenv('STOCK_SYMBOL'),
    'outputsize': 'full',
    'datatype': 'csv',
    'apikey': os.getenv('ALPHA_VANTAGE_API_KEY')
}

# send a GET request to the API
response = requests.get(url, params=params)

# create a Pandas DataFrame from the CSV response
data = pd.read_csv(io.BytesIO(response.content), parse_dates=['timestamp'])

# reverse the order of the data so it is in chronological order
data = data.iloc[::-1]

# Clean up the data
data = data.round(5)
data.drop(columns=['adjusted_close', 'volume', 'dividend_amount', 'split_coefficient'], inplace=True)
data.dropna(inplace=True)

# create csv file for history
now = datetime.now()
date_time = now.strftime("%Y-%m-%d_%H-%M-%S")
directory = "stock_data"
if not os.path.exists(directory):
    os.makedirs(directory)
filename = f"{directory}/{date_time}.csv"
data.to_csv(filename, index=False)

# Select features and target
x = data.drop(columns=['timestamp', 'open'])
y = np.where(data['open'].shift(-1) > data['open'], 1, 0)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train a random forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Define a function to make trading decisions
def make_trade_decision(data, model):
    # Clean up the data
    data = data.round(5)
   # data.drop(columns=['adjusted_close', 'volume', 'dividend_amount', 'split_coefficient'], inplace=True)
    data.dropna(inplace=True)
    
    # Select features
    data = data.drop(columns=['timestamp', 'open'])
    
    # Apply the trained model
    prediction = model.predict(data)
    
    print(prediction)

    # Return a trading signal based on the prediction
    if prediction == 1:
        return 'buy'
    else:
        return 'sell'

# Get the latest data
latest_data = data.iloc[-1:].copy()

# Make a trading decision
signal = make_trade_decision(latest_data, clf)

# Place a trade based on the trading signal
if signal == 'buy':
    # Place a buy order for the stock
    print('Placed a buy order')
elif signal == 'sell':
    # Place a sell order for the stock
    print('Placed a sell order')
else:
    # Do nothing
    print('No trade executed')