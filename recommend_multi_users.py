from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel
from pyspark.sql.functions import explode, col
import os
import sys
import glob
import shutil
import argparse

# ================= ENV =================
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
os.environ["SPARK_LOCAL_DIRS"] = "D:/spark_temp"

# ================= ARGUMENT =================
parser = argparse.ArgumentParser()
parser.add_argument("--top", type=int, default=5, help="Top N movies")
parser.add_argument("--users", type=int, default=10, help="Number of users")
args = parser.parse_args()

# ================= SPARK =================
spark = SparkSession.builder \
    .appName("Recommend Multiple Users") \
    .getOrCreate()

# ================= LOAD MODEL =================
model = ALSModel.load("model/my_als_model")

# ================= LOAD MOVIES =================
movies = spark.read.csv(
    "data/movie.csv",
    header=True,
    inferSchema=True
)
# ================= LOAD RATINGS =================
ratings = spark.read.csv(
    "data/rating.csv",
    header=True,
    inferSchema=True
)

users = ratings.select("userId").distinct()

# ================= CREATE USER LIST =================
users_df = spark.range(1, args.users + 1).withColumnRenamed("id", "userId")

# ================= RECOMMEND =================
recommendations = model.recommendForUserSubset(users_df, args.top)

# ================= EXPLODE =================
recommendations = recommendations \
    .withColumn("rec", explode("recommendations")) \
    .select(
        col("userId"),
        col("rec.movieId"),
        col("rec.rating").alias("score")
    )

# ================= JOIN TITLE =================
result = recommendations.join(movies, "movieId") \
    .select("userId", "title", "score") \
    .orderBy("userId", col("score").desc())

# ================= SAVE CSV =================
tmp_output = "output/tmp_recommend"
final_csv = f"output/recommend_multi_users_top{args.top}.csv"

result.coalesce(1) \
    .write \
    .mode("overwrite") \
    .option("header", "true") \
    .csv(tmp_output)

# ================= RENAME FILE =================
csv_file = glob.glob(f"{tmp_output}/part-*.csv")[0]
shutil.move(csv_file, final_csv)
shutil.rmtree(tmp_output)

print(f"✅ Saved recommendations to {final_csv}")

spark.stop()
