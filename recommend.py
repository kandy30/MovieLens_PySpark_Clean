from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import explode, col
import os
import sys
import argparse
import glob
import shutil


# ================= ENV =================
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
os.environ["SPARK_LOCAL_DIRS"] = "D:/spark_temp"


# ========== Gợi ý phim ==========
parser = argparse.ArgumentParser()
parser.add_argument("--user", type=int, default=1, help="User ID")
parser.add_argument("--top", type=int, default=5, help="Top N movies")
args = parser.parse_args()

# ========== Spark ==========
spark = SparkSession.builder \
    .appName("Movie Recommendation") \
    .getOrCreate()

# ========== Load model ==========
model = ALSModel.load("model/my_als_model")

# ========== Load movies ==========
movies = spark.read.csv(
    "data/movie.csv",
    header=True,
    inferSchema=True
)

# ========== Recommend ==========
users = spark.createDataFrame([(args.user,)], ["userId"])
recommendations = model.recommendForUserSubset(users, args.top)

# ========== Explode ==========
recommendations = recommendations \
    .withColumn("rec", explode("recommendations")) \
    .select(
        col("userId"),
        col("rec.movieId"),
        col("rec.rating").alias("score")
    )

# ========== Join title ==========
result = recommendations.join(movies, "movieId") \
    .select("userId", "title", "score")

print(f"\n🎬 Top {args.top} movie recommendations for user {args.user}:")
result.show(truncate=False)

# ========== Save to CSV ==========
output_dir = f"output/recommend_user_{args.user}_top{args.top}"

result.coalesce(1) \
    .write \
    .mode("overwrite") \
    .option("header", "true") \
    .csv(output_dir)

print(f"✅ Saved recommendation result to {output_dir}")
csv_file = glob.glob(f"{output_dir}/part-*.csv")[0]

# tên file mới
new_name = f"{output_dir}/recommend_user_{args.user}_top{args.top}.csv"

# đổi tên
shutil.move(csv_file, new_name)

print(f"📄 Renamed CSV to {new_name}")

spark.stop()