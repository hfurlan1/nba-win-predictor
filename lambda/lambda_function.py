import json
import boto3
import requests
import os
import time
from datetime import datetime, timezone

BUCKET = "nba-win-predictor-raw-hfurlan"
REGION = "us-east-1"
BASE_URL = "https://api.balldontlie.io/v1"


def fetch_all_games(api_key, season=2024):
    headers = {"Authorization": api_key}
    games = []
    cursor = None

    while True:
        params = {"seasons[]": season, "per_page": 100}
        if cursor:
            params["cursor"] = cursor

        response = requests.get(f"{BASE_URL}/games", headers=headers, params=params)

        print(f"Status: {response.status_code}, games so far: {len(games)}")

        if response.status_code == 429:
            print("Rate limited, waiting 10 seconds...")
            time.sleep(10)
            continue

        if response.status_code != 200:
            raise Exception(f"API error {response.status_code}: {response.text}")

        data = response.json()
        games.extend(data["data"])

        cursor = data["meta"].get("next_cursor")
        if not cursor:
            break

        time.sleep(1)

    return games


def lambda_handler(event, context):
    api_key = os.environ["BALLDONTLIE_API_KEY"]

    print("Fetching games from balldontlie...")
    games = fetch_all_games(api_key, season=2024)
    print(f"Fetched {len(games)} games")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    key = f"bronze/game_logs/raw_{timestamp}.json"

    s3 = boto3.client("s3", region_name=REGION)
    s3.put_object(
        Bucket=BUCKET,
        Key=key,
        Body=json.dumps(games),
        ContentType="application/json"
    )

    print(f"Uploaded to s3://{BUCKET}/{key}")
    return {"statusCode": 200, "body": f"Uploaded {len(games)} games to {key}"}