"""
Test Script for NBA Win Predictor
==================================
Sends a sample prediction request to the live application.

Requirements: pip install requests
Usage: python test_project.py
"""

import requests
import sys

API_URL = "http://54.147.89.206:8000"

def test_health():
    print("Testing API health...")
    resp = requests.get(f"{API_URL}/", timeout=30)
    if resp.status_code != 200:
        print(f"FAIL: Health check returned {resp.status_code}")
        sys.exit(1)
    print(f"OK: {resp.json()['status']}")

def test_teams():
    print("Testing /teams endpoint...")
    resp = requests.get(f"{API_URL}/teams", timeout=30)
    if resp.status_code != 200:
        print(f"FAIL: /teams returned {resp.status_code}")
        sys.exit(1)
    teams = resp.json()["teams"]
    print(f"OK: {len(teams)} teams available")

def test_team_stats():
    print("Testing /team-stats endpoint...")
    resp = requests.get(f"{API_URL}/team-stats/Boston Celtics", timeout=30)
    if resp.status_code != 200:
        print(f"FAIL: /team-stats returned {resp.status_code}")
        sys.exit(1)
    stats = resp.json()
    print(f"OK: Boston Celtics stats loaded — Win%: {stats['rolling_win_pct']}")

def test_predict():
    print("Testing /predict endpoint...")
    payload = {
        "home_rolling_win_pct": 0.8,
        "away_rolling_win_pct": 0.4,
        "home_rolling_avg_score": 118.0,
        "away_rolling_avg_score": 108.0,
        "home_rolling_avg_allowed": 110.0,
        "away_rolling_avg_allowed": 115.0,
        "home_days_rest": 2,
        "away_days_rest": 1
    }
    resp = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
    if resp.status_code != 200:
        print(f"FAIL: /predict returned {resp.status_code}")
        sys.exit(1)
    result = resp.json()
    print(f"Prediction: {result['prediction']} team wins")
    print(f"Home win probability: {result['home_win_probability']}%")
    print(f"Away win probability: {result['away_win_probability']}%")
    print("PASS")

if __name__ == "__main__":
    test_health()
    test_teams()
    test_team_stats()
    test_predict()
    print("\nAll tests passed. Exit code 0.")
    sys.exit(0)