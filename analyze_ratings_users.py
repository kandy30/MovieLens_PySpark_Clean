from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, count, avg, explode, split, when
)
import os
import sys
import glob
import shutil

# ================= ENV =================
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
os.environ["SPARK_LOCAL_DIRS"] = "D:/spark_temp"

# ================= SPARK =================
spark = SparkSession.builder \
    .appName("Analyze Ratings and Users") \
    .getOrCreate()

# ================= LOAD DATA =================
ratings = spark.read.csv(
    "data/rating.csv",
    header=True,
    inferSchema=True
)

movies = spark.read.csv(
    "data/movie.csv",
    header=True,
    inferSchema=True
)

# ================= HELPER SAVE CSV =================
def save_csv(df, output_name):
    tmp = f"output/tmp_{output_name}"
    final = f"output/{output_name}.csv"

    df.coalesce(1) \
      .write \
      .mode("overwrite") \
      .option("header", "true") \
      .csv(tmp)

    csv_file = glob.glob(f"{tmp}/part-*.csv")[0]
    shutil.move(csv_file, final)
    shutil.rmtree(tmp)

    print(f"✅ Saved {final}")

# =================================================
# 1️⃣ TOP 10 PHIM ĐƯỢC RATING NHIỀU NHẤT
# =================================================
top_movies_by_count = ratings.groupBy("movieId") \
    .agg(count("*").alias("num_ratings")) \
    .orderBy(col("num_ratings").desc()) \
    .limit(10) \
    .join(movies, "movieId") \
    .select("movieId", "title", "num_ratings")

top_movies_by_count.show(truncate=False)
save_csv(top_movies_by_count, "top_10_movies_most_rated")

# =================================================
# 2️⃣ TOP 10 PHIM CÓ RATING TRUNG BÌNH CAO NHẤT
# =================================================
top_movies_by_avg = ratings.groupBy("movieId") \
    .agg(
        count("*").alias("num_ratings"),
        avg("rating").alias("avg_rating")
    ) \
    .filter(col("num_ratings") >= 50) \
    .orderBy(col("avg_rating").desc()) \
    .limit(10) \
    .join(movies, "movieId") \
    .select("movieId", "title", "num_ratings", "avg_rating")

top_movies_by_avg.show(truncate=False)
save_csv(top_movies_by_avg, "top_10_movies_highest_rating")

# =================================================
# 3️⃣ TOP 10 USER HOẠT ĐỘNG NHIỀU NHẤT
# =================================================
top_users = ratings.groupBy("userId") \
    .agg(count("*").alias("num_ratings")) \
    .orderBy(col("num_ratings").desc()) \
    .limit(10)

top_users.show()
save_csv(top_users, "top_10_active_users")

# =================================================
# 4️⃣ PHÂN BỐ RATING
# =================================================
rating_distribution = ratings.groupBy("rating") \
    .agg(count("*").alias("count")) \
    .orderBy("rating")

rating_distribution.show()
save_csv(rating_distribution, "rating_distribution")

# =================================================
# ================== NÂNG CAO =====================
# =================================================

movie_ratings = ratings.join(movies, "movieId")

# =================================================
# 5️⃣ TOP GENRES ĐƯỢC YÊU THÍCH NHẤT
# =================================================
top_genres = movie_ratings \
    .withColumn("genre", explode(split(col("genres"), "\\|"))) \
    .groupBy("genre") \
    .agg(count("*").alias("num_ratings")) \
    .orderBy(col("num_ratings").desc())

top_genres.show(10, truncate=False)
save_csv(top_genres, "top_genres")

# =================================================
# 6️⃣ COLD-START USERS & MOVIES
# =================================================
cold_users = ratings.groupBy("userId") \
    .agg(count("*").alias("num_ratings")) \
    .filter(col("num_ratings") < 5)

cold_movies = ratings.groupBy("movieId") \
    .agg(count("*").alias("num_ratings")) \
    .filter(col("num_ratings") < 5) \
    .join(movies, "movieId") \
    .select("movieId", "title", "num_ratings")

save_csv(cold_users, "cold_start_users")
save_csv(cold_movies, "cold_start_movies")

# =================================================
# 7️⃣ PHÂN LOẠI HÀNH VI USER
# =================================================
user_behavior = ratings.groupBy("userId") \
    .agg(count("*").alias("num_ratings")) \
    .withColumn(
        "user_type",
        when(col("num_ratings") >= 100, "Very Active")
        .when(col("num_ratings") >= 50, "Active")
        .otherwise("Passive")
    )

user_behavior.groupBy("user_type").count().show()
save_csv(user_behavior, "user_behavior")

# =================================================
# 8️⃣ POPULARITY-BASED BASELINE
# =================================================
popular_movies = ratings.groupBy("movieId") \
    .agg(
        count("*").alias("num_ratings"),
        avg("rating").alias("avg_rating")
    ) \
    .filter(col("num_ratings") >= 50) \
    .orderBy(col("avg_rating").desc()) \
    .join(movies, "movieId") \
    .select("movieId", "title", "num_ratings", "avg_rating")

popular_movies.show(10, truncate=False)
save_csv(popular_movies, "popular_movies_baseline")

# ================= DONE =================
spark.stop()
print("🎉 All analysis completed. Results saved in 'output/'")
