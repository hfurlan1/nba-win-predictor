# Databricks notebook source
import boto3
import json
import pandas as pd
from pyspark.sql import SparkSession

# Download data from S3 using boto3
s3 = boto3.client(
    "s3",
    region_name="us-east-1",
    aws_access_key_id="API_KEY",
    aws_secret_access_key="SECRET_KEY"
)

# Get latest file
response = s3.list_objects_v2(
    Bucket="nba-win-predictor-raw-hfurlan",
    Prefix="bronze/game_logs/"
)
files = sorted(response["Contents"], key=lambda x: x["LastModified"], reverse=True)
latest_key = files[0]["Key"]
print(f"Reading: {latest_key}")

# Download and parse
obj = s3.get_object(Bucket="nba-win-predictor-raw-hfurlan", Key=latest_key)
games = json.loads(obj["Body"].read().decode("utf-8"))
print(f"Games loaded: {len(games)}")

# Convert to Spark DataFrame
df_raw = spark.createDataFrame(pd.DataFrame(games))
print(f"Records in DataFrame: {df_raw.count()}")
df_raw.printSchema()

# COMMAND ----------

# Flatten nested team structs and write to Delta bronze table
from pyspark.sql.functions import col

df_bronze = df_raw.select(
    col("id").alias("game_id"),
    col("date"),
    col("season"),
    col("status"),
    col("postseason"),
    col("postponed"),
    col("home_team_score"),
    col("visitor_team_score"),
    col("home_q1"), col("home_q2"), col("home_q3"), col("home_q4"),
    col("visitor_q1"), col("visitor_q2"), col("visitor_q3"), col("visitor_q4"),
    col("home_team.id").alias("home_team_id"),
    col("home_team.full_name").alias("home_team_name"),
    col("home_team.abbreviation").alias("home_team_abbr"),
    col("home_team.conference").alias("home_conference"),
    col("home_team.division").alias("home_division"),
    col("visitor_team.id").alias("visitor_team_id"),
    col("visitor_team.full_name").alias("visitor_team_name"),
    col("visitor_team.abbreviation").alias("visitor_team_abbr"),
    col("visitor_team.conference").alias("visitor_conference"),
    col("visitor_team.division").alias("visitor_division"),
)

# Write to Delta table
df_bronze.write.format("delta").mode("overwrite").saveAsTable("nba_bronze_game_logs")

print(f"Bronze table written: {df_bronze.count()} records")
df_bronze.show(3)

# COMMAND ----------

from pyspark.sql.functions import col, when, to_date

# Read from bronze
df_bronze = spark.table("nba_bronze_game_logs")

# Silver: filter to only completed games, add winner column
df_silver = df_bronze.filter(
    (col("status") == "Final") &
    (col("postponed") == False) &
    (col("home_team_score").isNotNull()) &
    (col("visitor_team_score").isNotNull())
).withColumn(
    "home_team_won",
    when(col("home_team_score") > col("visitor_team_score"), 1).otherwise(0)
).withColumn(
    "total_points",
    col("home_team_score") + col("visitor_team_score")
).withColumn(
    "point_differential",
    col("home_team_score") - col("visitor_team_score")
).withColumn(
    "date", to_date(col("date"))
).orderBy("date")

# Write to Delta silver table
df_silver.write.format("delta").mode("overwrite").saveAsTable("nba_silver_game_logs")

print(f"Silver table written: {df_silver.count()} records")
df_silver.select(
    "game_id", "date", "home_team_name", "visitor_team_name",
    "home_team_score", "visitor_team_score", "home_team_won"
).show(5)

# COMMAND ----------

from pyspark.sql import Window
from pyspark.sql.functions import (
    col, avg, lit, lag, datediff,
    when, round
)

from pyspark.sql import Window
from pyspark.sql.functions import (
    col, avg, sum, count, lag, datediff, 
    when, round, row_number
)

df_silver = spark.table("nba_silver_game_logs")

# Build a unified view: one row per team per game
# Home team perspective
df_home = df_silver.select(
    col("game_id"),
    col("date"),
    col("home_team_id").alias("team_id"),
    col("home_team_name").alias("team_name"),
    col("home_team_abbr").alias("team_abbr"),
    col("home_team_score").alias("team_score"),
    col("visitor_team_score").alias("opponent_score"),
    col("home_team_won").alias("won"),
    col("visitor_team_id").alias("opponent_id"),
    col("visitor_team_name").alias("opponent_name"),
    lit(1).alias("is_home")
)

# Visitor team perspective
df_away = df_silver.select(
    col("game_id"),
    col("date"),
    col("visitor_team_id").alias("team_id"),
    col("visitor_team_name").alias("team_name"),
    col("visitor_team_abbr").alias("team_abbr"),
    col("visitor_team_score").alias("team_score"),
    col("home_team_score").alias("opponent_score"),
    (1 - col("home_team_won")).alias("won"),
    col("home_team_id").alias("opponent_id"),
    col("home_team_name").alias("opponent_name"),
    lit(0).alias("is_home")
)

from pyspark.sql.functions import lit
df_home = df_silver.select(
    col("game_id"), col("date"),
    col("home_team_id").alias("team_id"),
    col("home_team_name").alias("team_name"),
    col("home_team_abbr").alias("team_abbr"),
    col("home_team_score").alias("team_score"),
    col("visitor_team_score").alias("opponent_score"),
    col("home_team_won").alias("won"),
    col("visitor_team_id").alias("opponent_id"),
    col("visitor_team_name").alias("opponent_name"),
    lit(1).alias("is_home")
)

df_away = df_silver.select(
    col("game_id"), col("date"),
    col("visitor_team_id").alias("team_id"),
    col("visitor_team_name").alias("team_name"),
    col("visitor_team_abbr").alias("team_abbr"),
    col("visitor_team_score").alias("team_score"),
    col("home_team_score").alias("opponent_score"),
    (1 - col("home_team_won")).alias("won"),
    col("home_team_id").alias("opponent_id"),
    col("home_team_name").alias("opponent_name"),
    lit(0).alias("is_home")
)

df_all = df_home.union(df_away).orderBy("team_id", "date")

# Rolling window: last 10 games per team
team_window = Window.partitionBy("team_id").orderBy("date").rowsBetween(-10, -1)

df_features = df_all.withColumn(
    "rolling_win_pct", round(avg("won").over(team_window), 3)
).withColumn(
    "rolling_avg_score", round(avg("team_score").over(team_window), 1)
).withColumn(
    "rolling_avg_allowed", round(avg("opponent_score").over(team_window), 1)
).withColumn(
    "days_rest", datediff(col("date"), lag("date", 1).over(
        Window.partitionBy("team_id").orderBy("date")
    ))
)

# Drop rows where rolling stats are null (first few games of season)
df_gold = df_features.filter(col("rolling_win_pct").isNotNull())

df_gold.write.format("delta").mode("overwrite").saveAsTable("nba_gold_team_features")

print(f"Gold table written: {df_gold.count()} records")
df_gold.select(
    "date", "team_name", "opponent_name", "is_home",
    "won", "rolling_win_pct", "rolling_avg_score", 
    "rolling_avg_allowed", "days_rest"
).show(5)

# COMMAND ----------


dbutils.secrets.listScopes()