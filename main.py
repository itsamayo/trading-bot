import pandas as pd
import numpy as np
import io
import os

from data_fetch import fetch_historical_data
from discord import send_discord_message
from place_order import place_trade_order, get_last_price
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# set up your env variables
STOCK_SYMBOL = os.getenv('STOCK_SYMBOL')
HEADERS = {'Content-type': 'application/json'}
ACCURACY_THRESHOLD = os.getenv('ACCURACY_THRESHOLD')

# fetch historical data
historical_data = fetch_historical_data()

# create a Pandas DataFrame from the CSV response
data = pd.read_csv(io.BytesIO(historical_data), parse_dates=['timestamp'])

# reverse the order of the data so it is in chronological order
data = data.iloc[::-1]

# clean up the data
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

# select features and target
x = data.drop(columns=['timestamp', 'open'])
y = np.where(data['open'].shift(-1) > data['open'], 1, 0)

# split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# train a random forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(x_train, y_train)

# get the latest data
latest_data = data.iloc[-1:].copy()

# determine accuracy
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

# define a function to make trading decisions
def make_trade_decision(data, model):
    # clean up the data
    data = data.round(5)   
    data.dropna(inplace=True)
    
    # select features
    data = data.drop(columns=['timestamp', 'open'])
    
    # apply the trained model
    prediction = model.predict(data)
    
    # return a trading signal based on the prediction
    if prediction == 1:
        return 'buy'
    else:
        return 'sell'

# main entry point
def main():
    # make a trading decision
    signal = make_trade_decision(latest_data, clf)

    # get prediction accuracy
    accuracy = define_accuracy(data)
    accuracy_perc = int(accuracy*100)

    # decide if we want to exit early based on accuracy or continue on to attempt a trade
    if accuracy_perc > int(ACCURACY_THRESHOLD):            
        last_price = get_last_price()
        if last_price == 0:
            send_discord_message(f':x: **UNAVAILABLE**: Looks like the market is either closed or **{STOCK_SYMBOL}** is not available right now')
            return
        if signal == 'buy':
            # attempt to place a buy order for the stock        
            send_discord_message(f':green_circle:  **BUY**: Latest **{STOCK_SYMBOL}** stock price: +-${last_price:.2f}')        
            place_trade_order(signal, last_price)
        elif signal == 'sell':
            # attempt to place a sell order for the stock        
            send_discord_message(f':red_circle:  **SELL**: Latest **{STOCK_SYMBOL}** stock price: +-${last_price:.2f}')
            place_trade_order(signal, last_price)
        else:
            # do nothing -- we should never get here
            send_discord_message(f'**OOPS**: Not sure how we got here but this should never happen')
    else:
        send_discord_message(f':octagonal_sign: **STOP**: Prediction accuracy fell below the configured accuracy threshold of **{ACCURACY_THRESHOLD}%** with an accuracy of **{accuracy_perc:.0f}%**')

main()