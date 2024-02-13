# Alpaca Trading Bot 💰
## NB - for learning purposes only, do not use this to trade with real money

## Details
A simple trading bot, that makes use of:
- The Alpha Advantage API to get historical stock data for a specified symbol
- The Alpaca Trading API to make market buy and sell orders
- A combination of `scikit-learn` `RandomForestClassifier` and `LogisticRegression` machine learning algorithms are used to make trading decisions
- Buy orders are configured to buy at a value of `(total buying power / buying power divider) / stock price` (configured in the .env file)
- Sell orders are configured to sell all currently held stock
- Orders are only placed if the prediction accuracy has a confidence level above the specified percentage (configured in the .env file)
- On bot run, messages are sent to a specified discord webhook at different checkpoints (configured in the .env file)

## Usage
Requires at least Python 3.7

Install dependencies
```
$ pip install -r requirements.txt
```
Configure a `.env` file by using the `.example_env` file and replacing all values with your own
1. You will need to create an [Alpha Advantage](https://www.alphavantage.co/) account to get an API key
2. You will need to create an [Alpaca](https://alpaca.markets/) account to get an API key and Secret
3. You will need a discord server that you can create webhooks for
```
$ cp .example.env .env
```
Run the bot
```
$ python main.py
```
## Continuous Deployment
This project is configured to deploy the bot to the configured server and send a message to the configured Discord webhook via Github Actions - 
*note* - this requires configuring the necessary secrets in the repo