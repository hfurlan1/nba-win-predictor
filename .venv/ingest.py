from dotenv import load_dotenv
load_dotenv()

import json
import boto3
import os
from datetime import datetime, timezone
from nba_api.stats.endpoints import leaguegamelog

BUCKET = "nba-win-predictor-raw-hfurlan"
REGION = "us-east-1"

def fetch_and_upload():
    print("Fetching game log from nba_api...")
    gamelog = leaguegamelog.LeagueGameLog(
        season="2024-25",
        season_type_all_star="Regular Season"
    )
    data = gamelog.get_dict()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    key = f"bronze/game_logs/raw_{timestamp}.json"

    s3 = boto3.client(
        "s3",
        region_name="us-east-1",
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"]
    )

    s3.put_object(
        Bucket=BUCKET,
        Key=key,
        Body=json.dumps(data),
        ContentType="application/json"
    )

    print(f"Uploaded to s3://{BUCKET}/{key}")

if __name__ == "__main__":
    fetch_and_upload()