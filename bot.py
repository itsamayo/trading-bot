import pandas as pd
import numpy as np
import requests
import io
import os
import json
import alpaca_trade_api as tradeapi
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# set up your API credentials
API_KEY = os.getenv('ALPACA_API_KEY')
SECRET_KEY = os.getenv('ALPACA_SECRET')
BASE_URL = os.getenv('ALPACA_ENDPOINT')
STOCK_SYMBOL = os.getenv('STOCK_SYMBOL')

# defined discord things
WEBHOOK_URL = os.getenv('DISCORD_WEBHOOK')
HEADERS = {'Content-type': 'application/json'}

# connect to the Alpaca API
api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')

# define the API URL and parameters
URL = 'https://www.alphavantage.co/query'
PARAMS = {
    'function': 'TIME_SERIES_DAILY_ADJUSTED',
    'symbol': STOCK_SYMBOL,
    'outputsize': 'full',
    'datatype': 'csv',
    'apikey': os.getenv('ALPHA_VANTAGE_API_KEY')
}

# send a GET request to the API
response = requests.get(URL, params=PARAMS)

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

# Define a function to determine accuracy
def define_accuracy(data):
    # clean up the data
    data['open'] = data['open'].apply(lambda x: round(x, 5))
    data['high'] = data['high'].apply(lambda x: round(x, 5))
    data['low'] = data['low'].apply(lambda x: round(x, 5))
    data['close'] = data['close'].apply(lambda x: round(x, 5))

    # fill missing values with the mean
    data = data.fillna(data.mean())

    # separate features (X) and target variable (y)
    X = data.drop(['timestamp', 'close'], axis=1)
    y = np.where(data['close'].shift(-1) > data['close'], 1, 0)

    # split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # normalize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # train a logistic regression model
    model = LogisticRegression(random_state=0)
    model.fit(X_train, y_train)

    # make predictions on the test set
    y_pred = model.predict(X_test)

    # calculate the accuracy of the model
    accuracy = (y_pred == y_test).mean()
    return accuracy

# Define a function to make trading decisions
def make_trade_decision(data, model):
    # Clean up the data
    data = data.round(5)   
    data.dropna(inplace=True)
    
    # Select features
    data = data.drop(columns=['timestamp', 'open'])
    
    # Apply the trained model
    prediction = model.predict(data)
    
    # Return a trading signal based on the prediction
    if prediction == 1:
        return 'buy'
    else:
        return 'sell'
    
# discord message sender
def send_discord_message(msg):
    message = {
            'content': msg
        }
    requests.post(WEBHOOK_URL, headers=HEADERS, data=json.dumps(message))
    
# place a market buy/sell order
def place_trade_order(side_type):
    discord_message_subj = 'BOUGHT' if side_type == 'buy' else 'SOLD'
    try:
        symbol = STOCK_SYMBOL
        qty = 1
        order_type = 'market'
        side = side_type
        resp = api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type=order_type,
            time_in_force='gtc'
        )        
        send_discord_message(f'**{discord_message_subj}** {STOCK_SYMBOL} stock at ${last_price:.2f}')
        print(resp)
    except:            
        send_discord_message(f'**FAILED** {side_type} - check logs for details')
    
# Get the latest data
latest_data = data.iloc[-1:].copy()

# Make a trading decision
signal = make_trade_decision(latest_data, clf)

# Send a message about a decision
last_price = data['close'].iloc[-1]
if signal == 'buy':
    # Place a buy order for the stock    
    accuracy = define_accuracy(data)    
    send_discord_message(f'**BUY**: Latest {STOCK_SYMBOL} stock price: ${last_price:.2f} *(buy prediction accuracy: {accuracy:.2f})*')
    place_trade_order(signal)
elif signal == 'sell':
    # Place a sell order for the stock
    accuracy = define_accuracy(data)
    send_discord_message(f'**SELL**: Latest {STOCK_SYMBOL} stock price: ${last_price:.2f} *(sell prediction accuracy: {accuracy:.2f})*')    
    place_trade_order(signal)
else:
    # Do nothing (not sure this should ever get here)
    send_discord_message(f'**OOPS**: not sure how we got here but this should never happen')