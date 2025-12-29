from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.ml.recommendation import ALS

# 1. Spark
spark = SparkSession.builder \
    .appName("Train_ALS_Local") \
    .master("local[2]") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.hadoop.fs.file.impl",
            "org.apache.hadoop.fs.LocalFileSystem") \
    .config("spark.hadoop.fs.file.impl.disable.cache", "true") \
    .config("spark.hadoop.fs.permissions.enabled", "false") \
    .getOrCreate()


# 2. Schema
ratings_schema = StructType([
    StructField("userId", IntegerType(), True),
    StructField("movieId", IntegerType(), True),
    StructField("rating", FloatType(), True),
    StructField("timestamp", LongType(), True)
])

# 3. Load data (LOCAL)
ratings = spark.read.csv(
    "data/rating.csv",   # ⬅️ kiểm tra đúng tên file
    header=True,
    schema=ratings_schema
).select("userId", "movieId", "rating")

# 4. Sample để giảm RAM
ratings = ratings.sample(fraction=0.3, seed=42)

# 5. Train ALS
als = ALS(
    userCol="userId",
    itemCol="movieId",
    ratingCol="rating",
    coldStartStrategy="drop",
    nonnegative=True,
    maxIter=5,
    rank=5,
    regParam=0.1
)

print("🚀 Đang train ALS...")
model = als.fit(ratings)

# 6. Save model
model.write().overwrite().save("model/my_als_model")

print("✅ Train xong & lưu model thành công!")
spark.stop()
