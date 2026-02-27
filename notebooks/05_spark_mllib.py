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
from pyspark.sql import DataFrame, SparkSession

LOGS_DIR = Path("/logs")
OUTPUT_DIR = Path("/output")
MODELS_DIR = OUTPUT_DIR / "models"
SAVE_MODEL_PATH = MODELS_DIR / "logistic_regression"
METRICS_PATH = OUTPUT_DIR / "metrics"
MODEL_COMPARISONS_PATH = METRICS_PATH / "model_comparison_spark.csv"

# ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_PATH.mkdir(parents=True, exist_ok=True)

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


def log_section(title: str) -> None:
    """Log a formatted section header.

    Args:
        title (str): Section title to display in logs.
    """
    logger.info("=" * 64)
    logger.info(title)
    logger.info("=" * 64)


def df_show_to_string(
    df: DataFrame, n: int = 10, truncate_int: int = 0, vertical: bool = False
) -> str:
    """Capture the output of df.show() as a string.

    Args:
        df (DataFrame): Spark DataFrame.
        n (int, optional): Number of rows to display. Defaults to 10.
        truncate (bool, optional): Max column width (False = no truncation). Defaults to 0.
        vertical (bool, optional): Vertical display mode. Defaults to False.

    Returns:
        str: String representation of df.show().
    """
    return "\n" + df._jdf.showString(n, truncate_int, vertical)


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

logger.info("Column 'Churn_num' added")

# ===
# DataFrame analysis
# ===

log_section("Schema")
logger.info("\n" + df._jdf.schema().treeString())

log_section("First 5 lines:")
logger.info(df_show_to_string(df, n=5))

log_section("Dimensions:")
size_df = df.count()
nb_columns = len(df.columns)
logger.info(f"Total number of lines: {size_df}")
logger.info(f"Total number of columns: {nb_columns}")

log_section("Statistics:")
logger.info(df_show_to_string(df.describe()))

log_section("Number of null per column:")
null_df = df.select(
    [F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c) for c in df.columns]
)
logger.info(df_show_to_string(null_df))

# ===
# Target Analysis
# ===

log_section("Distribution 'Churn':")
logger.info(df_show_to_string(df.groupBy("Churn").count()))

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

log_section("Split train / test:")
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

logger.info("Churn distribution train set:")
logger.info(df_show_to_string(churn_distribution_train))

logger.info("Churn distribution test set:")
logger.info(df_show_to_string(churn_distribution_test))

# ===
# Fitting and evaluate models
# ===

log_section("Fitting and evaluate models:")

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
fitted_models = {}

for name, model in models.items():
    t0 = time.time()
    model_fit = model.fit(train_df)
    t1 = time.time()

    fitted_models[name] = model_fit

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

metrics_df = pd.DataFrame(metrics).sort_values("Recall Yes", ascending=False)
logger.info("\n" + metrics_df.to_string(index=False))

best_model_name = metrics_df.iloc[0]["Model"]
logger.info(f"--> Best Model: {best_model_name}")

best_model = fitted_models[best_model_name]

# ===
# Feature importance
# ===

log_section("Feature importance:")
df_before_scaling = train_df
for stage in best_model.stages[:-2]:
    df_before_scaling = stage.transform(df_before_scaling)
attrs = df_before_scaling.schema["features"].metadata["ml_attr"]["attrs"]

feature_names = [attr["name"] for attr_type in attrs for attr in attrs[attr_type]]

clf_model = best_model.stages[-1]
importances = clf_model.featureImportances.toArray()

fi_pairs = sorted(zip(feature_names, importances), key=lambda x: -x[1])

lines = []
lines.append("Top 10 feature importance RandomForestClassifier:")
lines.append(f"{'Feature':<50} | {'Importance':<10} |")
for fname, importance in fi_pairs[:10]:
    bar = "▮" * int(importance * 100)
    lines.append(f"{fname:<50} | {importance:<10.2f} | {bar}")

logger.info("\n" + "\n".join(lines))

# ===
# Save
# ===

log_section("Save pipelines:")
best_model.write().overwrite().save(SAVE_MODEL_PATH.as_posix())
logger.info(f"Model, path={SAVE_MODEL_PATH.as_posix()}")

log_section("Save Spark results:")
metrics_df.to_csv(MODEL_COMPARISONS_PATH.as_posix())
logger.info(f"Model comparisons, path={MODEL_COMPARISONS_PATH.as_posix()}")


spark.stop()
