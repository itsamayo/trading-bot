import requests
import os
import json

from dotenv import load_dotenv

load_dotenv()

# set up your env variables
WEBHOOK_URL = os.getenv('DISCORD_WEBHOOK')

# discord message sender
def send_discord_message(msg):
    try:        
        message = { 'content': msg }
        requests.post(WEBHOOK_URL, headers={'Content-type': 'application/json'}, data=json.dumps(message))
    except:
        print('failed to send discord message')