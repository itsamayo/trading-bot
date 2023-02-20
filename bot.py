import pandas as pd
import numpy as np
import requests
import io
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
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
   # data.drop(columns=['adjusted_close', 'volume', 'dividend_amount', 'split_coefficient'], inplace=True)
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

# Get the latest data
latest_data = data.iloc[-1:].copy()

# Make a trading decision
signal = make_trade_decision(latest_data, clf)

# Send a message about a decision
webhook_url = os.getenv('DISCORD_WEBHOOK')
headers = {'Content-type': 'application/json'}
last_price = data['close'].iloc[-1]
if signal == 'buy':
    # Place a buy order for the stock    
    accuracy = define_accuracy(data)
    print(f'BUY order potential at ${last_price:.2f} for', os.getenv('STOCK_SYMBOL'))    
    message = {
        'content': f'**BUY**: Latest ABBV stock price: ${last_price:.2f} *(buy prediction accuracy: {accuracy:.2f})*'
    }
    requests.post(webhook_url, headers=headers, data=json.dumps(message))
elif signal == 'sell':
    # Place a sell order for the stock
    accuracy = define_accuracy(data)
    print(f'SELL order potential at ${last_price:.2f} for', os.getenv('STOCK_SYMBOL'))
    message = {
        'content': f'**SELL**: Latest ABBV stock price: ${last_price:.2f} *(sell prediction accuracy: {accuracy:.2f})*'
    }
    requests.post(webhook_url, headers=headers, data=json.dumps(message))
else:
    # Do nothing (not sure this should ever get here)
    print('Nothing to execute')