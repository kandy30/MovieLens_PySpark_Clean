from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml.recommendation import ALSModel
from pyspark.sql.functions import explode, col
import os
from pyspark.ml.recommendation import ALS

os.environ["PYSPARK_PYTHON"] = r"D:\DA bigdata\MovieLens_PySpark\venv310\Scripts\python.exe"
os.environ["PYSPARK_DRIVER_PYTHON"] = r"D:\DA bigdata\MovieLens_PySpark\venv310\Scripts\python.exe"

spark = SparkSession.builder \
    .appName("MovieLens_Local") \
    .master("local[*]") \
    .config("spark.memory.fraction", "0.8") \
    .getOrCreate()

print("✅ Spark đã khởi động")

ratings_schema = StructType([
    StructField("userId", IntegerType(), True),
    StructField("movieId", IntegerType(), True),
    StructField("rating", FloatType(), True),
    StructField("timestamp", LongType(), True)
])

movies_schema = StructType([
    StructField("movieId", IntegerType(), True),
    StructField("title", StringType(), True),
    StructField("genres", StringType(), True)
])

ratings = spark.read.csv("data/rating.csv", header=True, schema=ratings_schema)
movies = spark.read.csv("data/movie.csv", header=True, schema=movies_schema)


model_path = "model/my_als_model"
#als_model = ALSModel.load(model_path)
als = ALS(
    userCol="userId",
    itemCol="movieId",
    ratingCol="rating",
    coldStartStrategy="drop",
    nonnegative=True
)

als_model = als.fit(ratings)
als_model.save("model/my_als_model")


print("✅ Load ALS model thành công")

def recommend_for_user(user_id, n=5):
    user_df = spark.createDataFrame([(user_id,)], ["userId"])

    recs = als_model.recommendForUserSubset(user_df, n)

    result = recs.withColumn("rec", explode(col("recommendations"))) \
        .select(col("rec.movieId"), col("rec.rating")) \
        .join(movies, "movieId") \
        .select("title", "rating") \
        .orderBy(col("rating").desc())

    print(f"\n🎬 Gợi ý cho user {user_id}")
    result.show(truncate=False)

    # ===== TEST CHẠY THỬ =====
if __name__ == "__main__":
    recommend_for_user(1)
    recommend_for_user(10)
