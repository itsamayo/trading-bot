# Alpaca Trading Bot 💰

## Details
A simple trading bot, that makes use of:
- The Alpha Advantage API to get historical stock data for a specified symbol
- The Alpaca Trading API to make market buy and sell order. 
- A combination of `scikit-learn` `RandomForestClassifier` and `LogisticRegression` machine learning algorithms are used to make trading decisions.
- Buy orders are configured to buy at a value of `(total buying power / 2) / stock price`
- Sell order are configured to sell all currently held stock
- Orders are only placed if the prediciton accuracy has a confidence level above the specified percentage (configured in the .env file)
- On bot run, messages are sent to a specified discord webhook (configured in the .env file)

## Set up
Requires at least Python 3.7

Install dependencies
```
$ pip install -r requirements.txt
```
Configure a `.env` file by using the `.example_env` file and replacing all values with your own
1. You will need to create an Alpha Advantage account to get an API key
2. You will need to create an Alpaca account to get an API key and Secret
3. You will need a discord server that you can create webhooks for
```
$ cp .example.env .env
```
Run the bot
```
$ python bot.py
```
