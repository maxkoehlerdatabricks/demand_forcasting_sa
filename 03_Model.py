# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Train hundreds or thousands of models at scale
# MAGIC *with minimal code changes!*
# MAGIC 
# MAGIC **Prerequisite: Make sure to run the data simulation notebook 01_Simulate_Data before running this notebook.**  
# MAGIC 
# MAGIC While the previous notebook *(02_Explore_Data)* demonstrated the benefits of Databricks' collaborative and interactive environment,  
# MAGIC in this final part, we can apply that very same approach to train multiple models in parallel with great speed and cost-effectiveness.  
# MAGIC 
# MAGIC Key highlights for this notebook:
# MAGIC - Pandas UDFs can take your single-node data science code, and distribute it across different keys (e.g. SKU)  
# MAGIC - Hyperopt can also perform hyperparameter tuning from within a Pandas UDF  
# MAGIC - MLflow can track and log all of your parameters, metrics, and artifacts - which can be loaded for later use

# COMMAND ----------

import os
import json
import datetime
from random import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as md

import seaborn


from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from statsmodels.tsa.statespace.sarimax import SARIMAX

import mlflow
import hyperopt
from hyperopt import hp, fmin, tpe, Trials, SparkTrials, STATUS_OK, space_eval
from hyperopt.pyll.base import scope
mlflow.autolog(disable=True)

import pyspark.sql.functions as F
from pyspark.sql.types import *

# COMMAND ----------

# DBTITLE 1,Let's have a look at our main dataset again
data_path = '/FileStore/tables/demand_forecasting_solution_accellerator/demand_df_delta/'
demand_df = spark \
  .read \
  .format("delta") \
  .load(data_path)

display(demand_df)

# COMMAND ----------

# DBTITLE 1,Set high-level parameters 
forecast_horizon = 40
covid_breakpoint = datetime.date(year=2020, month=3, day=1)

# COMMAND ----------

# DBTITLE 1,Modularize single-node logic from before into Python functions
def define_exo_variables(pdf: pd.DataFrame) -> pd.DataFrame:
  
  exo_df = pdf.assign(Week = pd.DatetimeIndex(pdf["Date"]).isocalendar().week.tolist()) 

  exo_df = exo_df \
    .assign(covid = np.where(pdf["Date"] >= np.datetime64(covid_breakpoint), 1, 0).tolist()) \
    .assign(christmas = np.where((exo_df["Week"] >= 51) & (exo_df["Week"] <= 52) , 1, 0).tolist()) \
    .assign(new_year = np.where((exo_df["Week"] >= 1) & (exo_df["Week"] <= 4)  , 1, 0).tolist()) \
    .set_index('Date')

  exo_df = exo_df[["covid", "christmas", "new_year" ]]
  exo_df = exo_df.asfreq(freq='W-MON')
  return exo_df

# COMMAND ----------

def split_train_score_data(data, forecast_horizon=forecast_horizon):
  
  is_history = [True] * (len(data) - forecast_horizon) + [False] * forecast_horizon
  train = data.iloc[is_history]
  score = data.iloc[~np.array(is_history)]
  
  return train, score

# COMMAND ----------

def build_and_tune_model(sku_pdf: pd.DataFrame) -> pd.DataFrame:
  """
  This function trains and tunes a model for each SKU and can be distributed as a Pandas UDF
  """
  # Always ensure proper ordering by Date
  sku_pdf.sort_values("Date", inplace=True)
  
  # Since we'll group the large Spark DataFrame by (Product, SKU)
  PRODUCT = sku_pdf["Product"].iloc[0]
  SKU = sku_pdf["SKU"].iloc[0]
    
  # Create univariate time series indexed by Date 
  demand_series = pd.Series(sku_pdf['Demand'].values, index=sku_pdf['Date'])
  demand_series = demand_series.asfreq(freq='W-MON')
  train_data, score_data = split_train_score_data(demand_series)
  
  exo_df = define_exo_variables(sku_pdf)
  train_exo, score_exo = split_train_score_data(exo_df) 
  
  # Search space for ARIMA parameters
  search_space = {
    'p': scope.int(hyperopt.hp.quniform('p', 0, 4, 1)),
    'd': scope.int(hyperopt.hp.quniform('d', 0, 4, 1)),
    'q': scope.int(hyperopt.hp.quniform('q', 0, 4, 1)) 
  }
  
  def _score(hparams):
    # SARIMAX requires a tuple of Python integers
    order_hparams = tuple([int(hparams[k]) for k in ("p", "d", "q")])
    model = SARIMAX(
      train_data, 
      exog=train_exo, 
      order=order_hparams, 
      seasonal_order=(0, 0, 0, 0)
    )
    fit = model.fit(disp=False)
    fcast = fit.predict(start = min(score_data.index), end = max(score_data.index), exog = score_exo)
    loss = np.power(score_data.to_numpy() - fcast.to_numpy(), 2).mean()
    return loss
    
  # Iterate over search space
  best_hparams = fmin(_score, search_space, algo=tpe.suggest, max_evals=3)
  return pd.DataFrame(
    {'Product':[PRODUCT], 'SKU':[SKU], 'best_hparams':[json.dumps(best_hparams)]}
  )

# COMMAND ----------

demand_pdf = demand_df.toPandas()
unique_skus = demand_df.select("SKU").dropDuplicates().toPandas()
n_unique_skus = len(unique_skus)

# COMMAND ----------

spark.conf.set("spark.sql.shuffle.partitions", n_unique_skus)

tuning_schema = StructType(
  [
    StructField('Product', StringType()),
    StructField('SKU', StringType()),
    StructField('best_hparams', StringType())
  ]
)

tuned_df = (
  demand_df
  .groupBy("Product", "SKU")
  .applyInPandas(build_and_tune_model, schema=tuning_schema)
)
tuned_df = tuned_df.cache() # cache tuning results, most likely will be re-using them!

# COMMAND ----------

display(tuned_df)

# COMMAND ----------

# DBTITLE 1,Finally, let's log the parameters, metrics, and model artifacts in an organized manner
# MAGIC %md
# MAGIC 
# MAGIC Although we could've simply returned the forecast results as the output of the Pandas UDF above, it can often be beneficial to break down our workflow into a few steps:
# MAGIC 1. Model training and tuning by SKU
# MAGIC 2. Parameter, metric, and model logging - *perhaps by Product* (so that results and artifacts are a bit more well-organized and easier to retrieve)
# MAGIC 3. Inference (i.e. running your forecasts)
# MAGIC 
# MAGIC Modularizing your workflow can be extremely helpful for ensuring reproducibility and debuggability!

# COMMAND ----------

tuned_pdf = tuned_df.toPandas()

# COMMAND ----------

json.loads(tuned_pdf["best_hparams"].iloc[0])

# COMMAND ----------

# DBTITLE 1,Set up MLflow experiment
# Create an experiment name, which must be unique and case sensitive
experiment_id = mlflow.create_experiment("/Users/pawarit.laosunthara@databricks.com/Part-Level Demand Forecasting")
experiment = mlflow.get_experiment(experiment_id)

print("Name: {}".format(experiment.name))
print("Experiment_id: {}".format(experiment.experiment_id))
print("Artifact Location: {}".format(experiment.artifact_location))
print("Tags: {}".format(experiment.tags))
print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))

# COMMAND ----------

class ProductModelWrapper(mlflow.pyfunc.PythonModel):

  def __init__(self):
    self.models = {}

  def add_model(self, sku, fit):
    self.models[sku] = fit

  def predict(self, context, sku, data, exog):
    if sku not in self.models.keys():
      raise KeyError(f"No model found for SKU: {sku}")
    return self.models.get("sku").predict(start=min(exog.index), end=max(exog.index), exog=exog)

# COMMAND ----------



# COMMAND ----------

def log_to_mlflow(product_pdf: pd.DataFrame, sku_pdf: pd.DataFrame) -> pd.DataFrame:
  """
  In case there are millions of SKUs, it might be overwhelming to log/retrieve data to/from the MLflow server for every individual SKU.
  Instead we can group the SKU models/artifacts by e.g. Product to create a more organized hierarchy 
  """
  
  # Always ensure proper ordering by Date
  sku_pdf.sort_values("Date", inplace=True)
  
  # Since we'll group the large Spark DataFrame by (Product, SKU)
  PRODUCT = sku_pdf["Product"].iloc[0]
  SKU = sku_pdf["SKU"].iloc[0]
    
  # Create univariate time series indexed by Date 
  demand_series = pd.Series(sku_pdf['Demand'].values, index=sku_pdf['Date'])
  demand_series = demand_series.asfreq(freq='W-MON')
  train_data, score_data = split_train_score_data(demand_series)
  
  exo_df = define_exo_variables(sku_pdf)
  train_exo, score_exo = split_train_score_data(exo_df) 

  for idx, sku_row in product_pdf.iterrows():
    
    PRODUCT = sku_row["Product"]
    SKU = sku_row["SKU"]
    hparams = json.loads(sku_row["best_hparams"])
    
    order_hparams = tuple([int(hparams[k]) for k in ("p", "d", "q")])
    model = SARIMAX(
      train_data, 
      exog=train_exo, 
      order=order_hparams, 
      seasonal_order=(0, 0, 0, 0)
    )
    fit = model.fit(disp=False)
    print(fit)
    
#   with mlflow.start_run(experiment_id = experiment, run_name = PRODUCT):
#     mlflow.log_param()
#     mlflow.log_metric('auc', auc_score)

# COMMAND ----------

tuned_pdf = tuned_pdf.loc[tuned_pdf["SKU"] == "SRR_IMME9B"]

# COMMAND ----------

# sku_lolz

# COMMAND ----------

log_to_mlflow(tuned_pdf, sku_lolz)
