import pandas as pd
import numpy as np
import requests
import io
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
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

# clean up the data
data['open'] = data['open'].apply(lambda x: round(x, 5))
data['high'] = data['high'].apply(lambda x: round(x, 5))
data['low'] = data['low'].apply(lambda x: round(x, 5))
data['close'] = data['close'].apply(lambda x: round(x, 5))

# drop unneeded columns
data = data.drop(['adjusted_close', 'volume', 'dividend_amount', 'split_coefficient'], axis=1)

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
print(f'Accuracy: {accuracy:.2f}')
