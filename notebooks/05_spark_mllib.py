# ===
# docker exec -it spark-master /opt/spark/bin/spark-submit --master spark://spark-master:7077 /notebooks/05_spark_mllib.py
# ===

import logging
import time
from pathlib import Path

import pandas as pd
import pyspark.sql.functions as F
from pyspark.ml import Pipeline
from pyspark.ml.classification import (
    GBTClassifier,
    LogisticRegression,
    RandomForestClassifier,
)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import (
    Imputer,
    OneHotEncoder,
    StandardScaler,
    StringIndexer,
    VectorAssembler,
)
from pyspark.sql import SparkSession

SAVE_MODEL_PATH = "/output/models/logistic_regression"
MODEL_COMPARISON_PATH = "/output/metrics/model_comparison_spark.csv"

# ===
# Logging
# ===


def set_up_logger(
    name: str, logger_file_path: Path = None, level=logging.DEBUG
) -> logging.Logger:
    """Set up a logger.

    Args:
        name (str): Name of the logger.
        log_path (Path, optional): Path of the logger file. Defaults to None.
        level (_type_, optional): Level of the logger. Defaults to logging.INFO.

    Returns:
        logging.Logger: A configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter(
        "{asctime} - {levelname} - {message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if logger_file_path:
        file_handler = logging.FileHandler(
            logger_file_path,
            mode="wt",
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


logger = set_up_logger(
    name="05_spark_mllib", logger_file_path="/logs/05_spark_mllib.log"
)


# ===
# Session initialization
# ===

builder: SparkSession.Builder = SparkSession.builder
spark = (
    builder.appName("tp_final_partB")
    .master("spark://spark-master:7077")
    .config("spark.executor.memory", "1g")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

logger.info("Spark Session created")

# ===
# Data Loading
# ===

df = (
    spark.read.option("header", "true")
    .option("inferSchema", "true")
    .option("sep", ",")
    .csv("/data/03_DONNEES.csv")
)

logger.info("Data loaded")

df = df.withColumn("Churn_num", F.when(F.col("Churn") == "Yes", 1).otherwise(0))

logger.info("Add column 'Churn_num'")

# ===
# DataFrame analysis
# ===

print(f"\n{'=' * 64}")
print("Schema:")
print(f"{'=' * 64}")
df.printSchema()

print(f"\n{'=' * 64}")
print("First 5 lines:")
print(f"{'=' * 64}")
df.show(5)

print(f"\n{'=' * 64}")
print("Dimensions:")
print(f"{'=' * 64}")
size_df = df.count()
print("Total number of lines: ", size_df)
print("Total number of columns: ", len(df.columns))

print(f"\n{'=' * 64}")
print("Statistics:")
print(f"{'=' * 64}")
df.describe().show()

print(f"\n{'=' * 64}")
print("Number of null per column:")
print(f"{'=' * 64}")
df.select(
    [F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c) for c in df.columns]
).show()

# ===
# Target Analysis
# ===

print(f"\n{'=' * 64}")
print("Distribution 'Churn':")
print(f"{'=' * 64}")
df.groupBy("Churn").count().show()

# ===
# Preprocessing Pipeline
# ===

imputer = Imputer(
    inputCols=[
        "SeniorCitizen",
        "tenure",
        "InternetCharges",
        "MonthlyCharges",
        "TotalCharges",
    ],
    outputCols=[
        "SeniorCitizen_imp",
        "tenure_imp",
        "InternetCharges_imp",
        "MonthlyCharges_imp",
        "TotalCharges_imp",
    ],
    strategy="median",
)

string_indexer = StringIndexer(
    inputCols=[
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
    ],
    outputCols=[
        "gender_idx",
        "Partner_idx",
        "Dependents_idx",
        "PhoneService_idx",
        "MultipleLines_idx",
        "InternetService_idx",
        "OnlineSecurity_idx",
        "OnlineBackup_idx",
        "DeviceProtection_idx",
        "TechSupport_idx",
        "StreamingTV_idx",
        "StreamingMovies_idx",
        "Contract_idx",
    ],
    handleInvalid="keep",
)

one_hot_encoder = OneHotEncoder(
    inputCols=[
        "gender_idx",
        "Partner_idx",
        "Dependents_idx",
        "PhoneService_idx",
        "MultipleLines_idx",
        "InternetService_idx",
        "OnlineSecurity_idx",
        "OnlineBackup_idx",
        "DeviceProtection_idx",
        "TechSupport_idx",
        "StreamingTV_idx",
        "StreamingMovies_idx",
        "Contract_idx",
    ],
    outputCols=[
        "gender_ohe",
        "Partner_ohe",
        "Dependents_ohe",
        "PhoneService_ohe",
        "MultipleLines_ohe",
        "InternetService_ohe",
        "OnlineSecurity_ohe",
        "OnlineBackup_ohe",
        "DeviceProtection_ohe",
        "TechSupport_ohe",
        "StreamingTV_ohe",
        "StreamingMovies_ohe",
        "Contract_ohe",
    ],
    dropLast=True,
)

vector_assembler = VectorAssembler(
    inputCols=[
        "SeniorCitizen_imp",
        "tenure_imp",
        "InternetCharges_imp",
        "MonthlyCharges_imp",
        "TotalCharges_imp",
        "gender_ohe",
        "Partner_ohe",
        "Dependents_ohe",
        "PhoneService_ohe",
        "MultipleLines_ohe",
        "InternetService_ohe",
        "OnlineSecurity_ohe",
        "OnlineBackup_ohe",
        "DeviceProtection_ohe",
        "TechSupport_ohe",
        "StreamingTV_ohe",
        "StreamingMovies_ohe",
        "Contract_ohe",
    ],
    outputCol="features",
    handleInvalid="skip",
)

standard_scaler = StandardScaler(
    inputCol="features",
    outputCol="features_scaled",
    withMean=True,
    withStd=True,
)

preprocessor = Pipeline(
    stages=[
        imputer,
        string_indexer,
        one_hot_encoder,
        vector_assembler,
        standard_scaler,
    ]
)

# ===
# Define Models
# ===

models = {
    "RandomForestClassifier": Pipeline(
        stages=preprocessor.getStages()
        + [
            RandomForestClassifier(
                featuresCol="features_scaled",
                labelCol="Churn_num",
                predictionCol="prediction",
                seed=42,
            )
        ]
    ),
    "LogisticRegression": Pipeline(
        stages=preprocessor.getStages()
        + [
            LogisticRegression(
                featuresCol="features_scaled",
                labelCol="Churn_num",
                predictionCol="prediction",
            )
        ]
    ),
    "GBTClassifier": Pipeline(
        stages=preprocessor.getStages()
        + [
            GBTClassifier(
                featuresCol="features_scaled",
                labelCol="Churn_num",
                predictionCol="prediction",
                seed=42,
            )
        ]
    ),
}

# ===
# Split train / test
# ===

print(f"\n{'=' * 64}")
print("Split train / test:")
print(f"{'=' * 64}")
train_df, test_df = df.randomSplit(weights=[0.7, 0.3], seed=42)

total_count_train = train_df.count()
total_count_test = test_df.count()

churn_distribution_train = (
    train_df.groupBy("Churn")
    .count()
    .withColumn(
        "percentage", F.round(F.col("count") / F.lit(total_count_train) * 100, 2)
    )
)
churn_distribution_test = (
    test_df.groupBy("Churn")
    .count()
    .withColumn(
        "percentage", F.round(F.col("count") / F.lit(total_count_test) * 100, 2)
    )
)

print("Churn distribution train set:")
churn_distribution_train.show()

print("\nChurn distribution test set:")
churn_distribution_test.show()

# ===
# Fitting and evaluate models
# ===

print(f"\n{'=' * 64}")
print("Metrics:")
print(f"{'=' * 64}")
evaluator_accuracy = MulticlassClassificationEvaluator(
    labelCol="Churn_num",
    predictionCol="prediction",
    metricName="accuracy",
)

evaluator_f1 = MulticlassClassificationEvaluator(
    labelCol="Churn_num",
    predictionCol="prediction",
    metricName="f1",
)

evaluator_precision = MulticlassClassificationEvaluator(
    labelCol="Churn_num",
    predictionCol="prediction",
    metricName="weightedPrecision",
)

evaluator_recall = MulticlassClassificationEvaluator(
    labelCol="Churn_num",
    predictionCol="prediction",
    metricName="weightedRecall",
)

metrics = []

for name, model in models.items():
    t0 = time.time()
    model_fit = model.fit(train_df)
    t1 = time.time()

    logger.info(f"Model {name} fit: time={t1 - t0}s")

    t2 = time.time()
    preds = model_fit.transform(test_df)

    confusion = preds.groupBy("Churn_num", "prediction").count()
    tp = preds.filter((F.col("Churn_num") == 1) & (F.col("prediction") == 1)).count()
    tn = preds.filter((F.col("Churn_num") == 0) & (F.col("prediction") == 0)).count()
    fp = preds.filter((F.col("Churn_num") == 0) & (F.col("prediction") == 1)).count()
    fn = preds.filter((F.col("Churn_num") == 1) & (F.col("prediction") == 0)).count()

    precision_yes = tp / (tp + fp) if (tp + fp) != 0 else 0
    precision_no = tn / (tn + fn) if (tn + fn) != 0 else 0
    recall_yes = tp / (tp + fn) if (tp + fn) != 0 else 0
    recall_no = tn / (tn + fp) if (tn + fp) != 0 else 0

    metrics.append(
        {
            "Model": name,
            "Accuracy": evaluator_accuracy.evaluate(preds),
            "Precision Yes": precision_yes,
            "Precision No": precision_no,
            "Precision Weighted": evaluator_precision.evaluate(preds),
            "Recall Yes": recall_yes,
            "Recall No": recall_no,
            "Recall Weighted": evaluator_recall.evaluate(preds),
            "F1 Yes": (
                2 * precision_yes * recall_yes / (precision_yes + recall_yes)
                if (precision_yes + recall_yes) != 0
                else 0
            ),
            "F1 No": (
                2 * precision_no * recall_no / (precision_no + recall_no)
                if (precision_no + recall_no) != 0
                else 0
            ),
            "F1 Weighted": evaluator_f1.evaluate(preds),
            "Time": t1 - t0,
        }
    )
    t3 = time.time()

    logger.info(f"Model {name} evaluate: time={t3 - t2}s")

print(
    f"{'Model':<30} | {'Accuracy':<10} | {'Precision Yes':<10} | {'Recall Yes':<10} | {'F1 Yes':<10} | {'Time':<10}"
)
for metric in metrics:
    print(
        f"{metric['Model']:<30} | {metric['Accuracy']:<10.6f} | {metric['Precision Yes']:<10.6f} | {metric['Recall Yes']:<10.6f} | {metric['F1 Yes']:<10.6f} | {metric['Time']:<10.6f}"
    )

metrics_df = pd.DataFrame(metrics)
best_model_name = metrics_df.sort_values("Recall Yes").loc[0, "Model"]
print(f"--> Best Model: {best_model_name}")

best_model = models[best_model_name]
print(type(best_model))

# ===
# Feature importance
# ===

print(f"\n{'=' * 64}")
print("Feature importance:")
print(f"{'=' * 64}")
# df_before_scaling = train_df
# for stage in best_model.stages[:-2]:
#     df_before_scaling = stage.transform(df_before_scaling)
# attrs = df_before_scaling.schema["features"].metadata["ml_attr"]["attrs"]

# feature_names = [attr["name"] for attr_type in attrs for attr in attrs[attr_type]]

# clf_model = best_model.stages[-1]
# importances = clf_model.featureImportances.toArray()

# fi_pairs = sorted(zip(feature_names, importances), key=lambda x: -x[1])

# print("Top 10 feature importance RandomForestClassifier:")
# print(f"{'feature':<30} | {'importance':<10} | {'bar'}")
# for fname, importance in fi_pairs[:10]:
#     bar = "▮" * int(importance * 100)
#     print(f"{fname:<30} | {importance:<10.2f} | {bar}")

# ===
# Save
# ===

print(f"\n{'=' * 64}")
print("Save pipelines:")
print(f"{'=' * 64}")
best_model.write().overwrite().save(SAVE_MODEL_PATH)
print("Model path: ", SAVE_MODEL_PATH)

print(f"\n{'=' * 64}")
print("Save Spark results:")
print(f"{'=' * 64}")
metrics_df.to_csv(MODEL_COMPARISON_PATH)
# data = {}
# data.update(results_reg)
# data.update(results_clf)
# json_path = EXOS_PATH / "spark_results.json"
# json_path.write_text(data=json.dumps(data), encoding="utf-8")
# print("Spark results path: ", json_path)


spark.stop()
