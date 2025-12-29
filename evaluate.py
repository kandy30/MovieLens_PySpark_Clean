from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel
from pyspark.ml.evaluation import RegressionEvaluator
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd

# ================= ENV =================
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
os.environ["SPARK_LOCAL_DIRS"] = "D:/spark_temp"

# ================= SPARK =================
spark = SparkSession.builder \
    .appName("Evaluate ALS RMSE") \
    .getOrCreate()

# ================= LOAD DATA =================
ratings = spark.read.csv(
    "data/rating.csv",
    header=True,
    inferSchema=True
)

# ================= SPLIT =================
train, test = ratings.randomSplit([0.8, 0.2], seed=42)

# ================= LOAD MODEL =================
model = ALSModel.load("model/my_als_model")

# ================= PREDICT =================
predictions = model.transform(test).dropna()

# ================= EVALUATE =================
evaluator = RegressionEvaluator(
    metricName="rmse",
    labelCol="rating",
    predictionCol="prediction"
)

rmse = evaluator.evaluate(predictions)
print(f"📊 RMSE = {rmse}")

# ================= SAVE CSV (PANDAS) =================
rmse_pd = pd.DataFrame({
    "metric": ["RMSE"],
    "value": [rmse]
})

os.makedirs("output", exist_ok=True)
rmse_pd.to_csv("output/rmse_als_model.csv", index=False)

# ================= PLOT =================
plt.figure()
plt.bar(rmse_pd["metric"], rmse_pd["value"])
plt.title("ALS Model RMSE")
plt.ylabel("RMSE")
plt.savefig("output/rmse_plot.png")
plt.show()

spark.stop()
