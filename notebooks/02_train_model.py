# Databricks notebook source
# MAGIC %pip install xgboost

# COMMAND ----------

from pyspark.sql.functions import col

# Read gold table
df_gold = spark.table("nba_gold_team_features")

# Separate home and away perspectives
df_home = df_gold.filter(col("is_home") == 1).select(
    col("game_id"),
    col("date"),
    col("team_id").alias("home_team_id"),
    col("team_name").alias("home_team_name"),
    col("won").alias("home_team_won"),
    col("rolling_win_pct").alias("home_rolling_win_pct"),
    col("rolling_avg_score").alias("home_rolling_avg_score"),
    col("rolling_avg_allowed").alias("home_rolling_avg_allowed"),
    col("days_rest").alias("home_days_rest"),
    col("opponent_id").alias("visitor_team_id"),
)

df_away = df_gold.filter(col("is_home") == 0).select(
    col("game_id"),
    col("rolling_win_pct").alias("away_rolling_win_pct"),
    col("rolling_avg_score").alias("away_rolling_avg_score"),
    col("rolling_avg_allowed").alias("away_rolling_avg_allowed"),
    col("days_rest").alias("away_days_rest"),
)

# Join home and away into one row per game
df_training = df_home.join(df_away, on="game_id", how="inner").filter(
    col("home_days_rest").isNotNull() &
    col("away_days_rest").isNotNull()
)

print(f"Training records: {df_training.count()}")
df_training.select(
    "date", "home_team_name", "home_team_won",
    "home_rolling_win_pct", "away_rolling_win_pct",
    "home_days_rest", "away_days_rest"
).show(5)

# COMMAND ----------

import mlflow
import mlflow.xgboost
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import xgboost as xgb
from mlflow.models.signature import infer_signature

# Convert to pandas
df_pd = df_training.select(
    "home_rolling_win_pct",
    "away_rolling_win_pct",
    "home_rolling_avg_score",
    "away_rolling_avg_score",
    "home_rolling_avg_allowed",
    "away_rolling_avg_allowed",
    "home_days_rest",
    "away_days_rest",
    "home_team_won"
).toPandas()

FEATURES = [
    "home_rolling_win_pct",
    "away_rolling_win_pct",
    "home_rolling_avg_score",
    "away_rolling_avg_score",
    "home_rolling_avg_allowed",
    "away_rolling_avg_allowed",
    "home_days_rest",
    "away_days_rest",
]
TARGET = "home_team_won"

X = df_pd[FEATURES]
y = df_pd[TARGET]

split_idx = int(len(df_pd) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

mlflow.set_experiment("/nba-win-predictor")

with mlflow.start_run(run_name="xgboost_v2"):
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss"
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    # Create signature
    signature = infer_signature(X_train, model.predict_proba(X_train))

    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 4)
    mlflow.log_param("learning_rate", 0.1)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("auc_roc", auc)

    mlflow.xgboost.log_model(
        model,
        name="model",
        signature=signature,
        input_example=X_train.iloc[:3]
    )

    print(f"Accuracy: {accuracy:.3f}")
    print(f"AUC-ROC:  {auc:.3f}")
    print("Model logged to MLflow with signature")

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Get the latest run
experiment = client.get_experiment_by_name("/nba-win-predictor")
runs = client.search_runs(experiment.experiment_id, order_by=["start_time DESC"], max_results=1)
latest_run = runs[0]
run_id = latest_run.info.run_id

print(f"Run ID: {run_id}")

# Register the model
model_uri = f"runs:/{run_id}/model"
registered = mlflow.register_model(model_uri, "nba-win-predictor")

print(f"Model registered: {registered.name} version {registered.version}")

# COMMAND ----------

import mlflow
import pickle
import boto3

# Load the registered model
model_uri = "models:/workspace.default.nba-win-predictor/1"
model = mlflow.xgboost.load_model(model_uri)

# Save as pickle
with open("/tmp/nba_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Upload to S3
s3 = boto3.client(
    "s3",
    region_name="us-east-1",
    aws_access_key_id="API_KEY",
    aws_secret_access_key="SECRET_KEY"
)

s3.upload_file("/tmp/nba_model.pkl", "nba-win-predictor-raw-hfurlan", "model/nba_model.pkl")
print("Model uploaded to S3")

# COMMAND ----------

import boto3
import json
from pyspark.sql.functions import col, max as spark_max

df_gold = spark.table("nba_gold_team_features")

# Get latest date per team
latest_dates = df_gold.groupBy("team_name").agg(spark_max("date").alias("max_date"))

# Join with aliases to avoid ambiguity
df_a = df_gold.alias("a")
df_b = latest_dates.alias("b")

df_latest = df_a.join(df_b,
    (col("a.team_name") == col("b.team_name")) &
    (col("a.date") == col("b.max_date"))
).select(
    col("a.team_name"),
    col("a.rolling_win_pct"),
    col("a.rolling_avg_score"),
    col("a.rolling_avg_allowed"),
    col("a.days_rest")
)

rows = df_latest.collect()
team_stats = {}
for row in rows:
    team_stats[row.team_name] = {
        "rolling_win_pct": row.rolling_win_pct,
        "rolling_avg_score": row.rolling_avg_score,
        "rolling_avg_allowed": row.rolling_avg_allowed,
        "days_rest": int(row.days_rest) if row.days_rest else 2
    }

print(f"Teams with stats: {len(team_stats)}")
print(list(team_stats.keys())[:5])

s3 = boto3.client(
    "s3",
    region_name="us-east-1",
    aws_access_key_id="API_KEY",
    aws_secret_access_key="SECRET_KEY"
)

s3.put_object(
    Bucket="nba-win-predictor-raw-hfurlan",
    Key="model/team_stats.json",
    Body=json.dumps(team_stats),
    ContentType="application/json"
)
print("Team stats uploaded to S3")