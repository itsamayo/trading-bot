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

# set up your env variables
API_KEY = os.getenv('ALPACA_API_KEY')
SECRET_KEY = os.getenv('ALPACA_SECRET')
BASE_URL = os.getenv('ALPACA_ENDPOINT')
STOCK_SYMBOL = os.getenv('STOCK_SYMBOL')
WEBHOOK_URL = os.getenv('DISCORD_WEBHOOK')
HEADERS = {'Content-type': 'application/json'}
ACCURACY_THRESHOLD = os.getenv('ACCURACY_THRESHOLD')

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

"""
THE MEAT of the bot for:
Data fetching
Data prepping
Feature and target determination
Splitting data into training and testing sets
Training of a Random Forest Classifier
"""

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
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train a random forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(x_train, y_train)

# Get the latest data
latest_data = data.iloc[-1:].copy()

"""
End of THE MEAT
"""

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
    x = data.drop(columns=['timestamp', 'open'])
    y = np.where(data['open'].shift(-1) > data['open'], 1, 0)

    # split the data into training and test sets
    acc_x_train, acc_x_test, acc_y_train, acc_y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # normalize the data
    scaler = StandardScaler()
    acc_x_train = scaler.fit_transform(acc_x_train)
    acc_x_test = scaler.transform(acc_x_test)

    # train a logistic regression model
    model = LogisticRegression(random_state=0)
    model.fit(acc_x_train, acc_y_train)

    # make predictions on the test set
    y_pred = model.predict(acc_x_test)

    # calculate the accuracy of the model
    accuracy = (y_pred == acc_y_test).mean()
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

# get current position
def get_current_position():
    shares_held_currently = 0
    portfolio = api.list_positions()    
    for position in portfolio:
        if position.symbol == STOCK_SYMBOL:
            shares_held_currently = position.qty
    return shares_held_currently
    
# place a market buy/sell order
def place_trade_order(signal, last_price):
    discord_message_subj = ':money_with_wings: BOUGHT' if signal == 'buy' else ':moneybag: SOLD'    
    qty = 0    
    account = api.get_account()

    # exit early if trading is blocked on the account
    if account.trading_blocked:
        send_discord_message(f'**BLOCKED**: Trading is currently unavailable')
        return
    
    # exit early if half of buying power wouldn't be enough to buy a single share
    if float(account.buying_power)/2 < last_price:
        send_discord_message(f':skull_crossbones: **BROKE ASS**: Not enough buying power to satisfy rules')

    # get a list of all of our positions to set shares_held_currently
    shares_held_currently = get_current_position()

    # for sell orders, we want to sell all owned shares, for buys we set that in the env variables
    if signal == 'sell':
        qty = shares_held_currently
    else:
        # for buy orders we want to buy for no more than half of what our current buying power allows
        qty = int((float(account.buying_power)/2)/float(last_price))
    
    if qty > 0:
        try:
            send_discord_message(f':rocket: Currently holding {shares_held_currently} {STOCK_SYMBOL} share/s')
            api.submit_order(
                symbol=STOCK_SYMBOL,
                qty=qty,
                side=signal,
                type='market',
                time_in_force='gtc'
            )
            send_discord_message(f'**{discord_message_subj}** {qty} {STOCK_SYMBOL} share/s at +-${last_price:.2f}')
        except:            
            send_discord_message(f':sob: **FAILED**{signal}: Check logs for details')
    else:
        if signal == 'sell':
            send_discord_message(f':pinching_hand: **NO POSITIONS**: Can\'t sell what you don\'t have')        

def run_trader_bot():
    # make a trading decision
    signal = make_trade_decision(latest_data, clf)

    # get prediction accuracy and decide if we want to exit early
    accuracy = define_accuracy(data)
    accuracy_perc = int(accuracy*100)

    if accuracy_perc > int(ACCURACY_THRESHOLD):
        # send a message about a decision
        last_price_raw = api.get_latest_quote(STOCK_SYMBOL)
        last_price = last_price_raw.ap
        if signal == 'buy':
            # attempt tp place a buy order for the stock        
            send_discord_message(f':green_circle:  **BUY**: Latest {STOCK_SYMBOL} stock price: +-${last_price:.2f}')        
            place_trade_order(signal, last_price)
        elif signal == 'sell':
            # attempt to place a sell order for the stock        
            send_discord_message(f':red_circle:  **SELL**: Latest {STOCK_SYMBOL} stock price: +-${last_price:.2f}')
            place_trade_order(signal, last_price)
        else:
            # do nothing -- we should never get here
            send_discord_message(f'**OOPS**: Not sure how we got here but this should never happen')
    else:
        send_discord_message(f':octagonal_sign: **STOP**: Prediction accuracy fell below the configured accuracy threshold of **{ACCURACY_THRESHOLD}%** with an accuracy of **{accuracy_perc:.0f}%**')

run_trader_bot()
