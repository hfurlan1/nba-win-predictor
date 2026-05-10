from dotenv import load_dotenv
load_dotenv()

import requests
import os

key = os.environ['BALLDONTLIE_API_KEY']
r = requests.get(
    'https://api.balldontlie.io/v1/games?seasons[]=2024&per_page=5',
    headers={'Authorization': key}
)
print(r.status_code)
print(r.json())