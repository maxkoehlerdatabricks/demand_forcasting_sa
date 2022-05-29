# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Train thousands of models at scale, any time
# MAGIC *while still using your preferred libraries and approaches*

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC *Prerequisite: Make sure to run the data simulation notebook 01_Simulate_Data before running this notebook.*
# MAGIC 
# MAGIC While the previous notebook *(02_Explore_Data)* demonstrated the benefits of Databricks' collaborative and interactive environment,  
# MAGIC in this final part, we can apply that very same approach to train multiple models in parallel with great speed and cost-effectiveness.  
# MAGIC 
# MAGIC Key highlights for this notebook:
# MAGIC - Pandas UDFs (user-defined functions) can take your single-node data science code, and distribute it across different keys (e.g. SKU)  
# MAGIC - Hyperopt can also perform hyperparameter tuning from within a Pandas UDF  

# COMMAND ----------

# DBTITLE 0,Import libraries
import json
import pickle

import datetime as dt
from random import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as md

import seaborn

from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_percentage_error

import mlflow
import hyperopt
from hyperopt import hp, fmin, tpe, Trials, SparkTrials, STATUS_OK, space_eval
from hyperopt.pyll.base import scope
mlflow.autolog(disable=True)

import pyspark.sql.functions as F
from pyspark.sql.types import *

# COMMAND ----------

# DBTITLE 0,Modularize single-node logic from before into Python functions
FORECAST_HORIZON = 40

def add_exo_variables(pdf: pd.DataFrame) -> pd.DataFrame:
  
  midnight = dt.datetime.min.time()
  timestamp = pdf["Date"].apply(lambda x: dt.datetime.combine(x, midnight))
  calendar_week = timestamp.dt.isocalendar().week
  
  # define flexible, custom logic for exogenous variables
  covid_breakpoint = dt.datetime(year=2020, month=3, day=1)
  enriched_df = (
    pdf
      .assign(covid = (timestamp >= covid_breakpoint).astype(float))
      .assign(christmas = ((calendar_week >= 51) & (calendar_week <= 52)).astype(float))
      .assign(new_year = ((calendar_week >= 1) & (calendar_week <= 4)).astype(float))
  )
  return enriched_df[["Date", "Product", "SKU", "Demand", "covid", "christmas", "new_year"]]

enriched_schema = StructType(
  [
    StructField('Date', DateType()),
    StructField('Product', StringType()),
    StructField('SKU', StringType()),
    StructField('Demand', FloatType()),
    StructField('covid', FloatType()),
    StructField('christmas', FloatType()),
    StructField('new_year', FloatType()),
  ]
)

def split_train_score_data(data, forecast_horizon=FORECAST_HORIZON):
  """
  - assumes data is sorted by date/time already
  - forecast_horizon in weeks
  """
  is_history = [True] * (len(data) - forecast_horizon) + [False] * forecast_horizon
  train = data.iloc[is_history]
  score = data.iloc[~np.array(is_history)]
  return train, score

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Let's have a look at our core dataset, this time with exogenous variables
# MAGIC 
# MAGIC This time, data for all SKUs is logically unified within a Spark DataFrame, allowing large-scale distributed processing.

# COMMAND ----------

demand_df = spark.read.table("demand_db.part_level_demand")
demand_df = demand_df.cache() # just for this example notebook

enriched_df = (
  demand_df
    .groupBy("Product")
    .applyInPandas(add_exo_variables, enriched_schema)
)
display(enriched_df)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ## Solution: High-Level Overview
# MAGIC 
# MAGIC #### Benefits
# MAGIC - Pure Python & Pandas: easy to develop, test
# MAGIC - Continue using your favorite libraries
# MAGIC - Simply assume you're working with a Pandas DataFrame for a single SKU
# MAGIC 
# MAGIC <img src="https://github.com/PawaritL/data-ai-world-tour-dsml-jan-2022/blob/main/pandas-udf-workflow.png?raw=true" width=40%>

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Build, tune ansd score a model per each SKU with Pandas UDFs

# COMMAND ----------

def build_tune_and_score_model(sku_pdf: pd.DataFrame) -> pd.DataFrame:
  """
  This function trains, tunes and scores a model for each SKU and can be distributed as a Pandas UDF
  """
  # Always ensure proper ordering and indexing by Date
  sku_pdf.sort_values("Date", inplace=True)
  complete_ts = sku_pdf.set_index("Date").asfreq(freq="W-MON")

  # Since we'll group the large Spark DataFrame by (Product, SKU)
  PRODUCT = sku_pdf["Product"].iloc[0]
  SKU = sku_pdf["SKU"].iloc[0]
  train_data, validation_data = split_train_score_data(complete_ts)
  exo_fields = ["covid", "christmas", "new_year"]


  # Evaluate model on the traing data set
  def evaluate_model(hyperopt_params):

        # SARIMAX requires a tuple of Python integers
        order_hparams = tuple([int(hyperopt_params[k]) for k in ("p", "d", "q")])

        # Training
        model = SARIMAX(
          train_data["Demand"], 
          exog=train_data[exo_fields], 
          order=order_hparams, 
          seasonal_order=(0, 0, 0, 0), # assume no seasonality in our example
          initialization_method="estimated",
          enforce_stationarity = False,
          enforce_invertibility = False
        )
        fitted_model = model.fit(disp=False, method='nm')

        # Validation
        fcast = fitted_model.predict(
          start=validation_data.index.min(), 
          end=validation_data.index.max(), 
          exog=validation_data[exo_fields]
        )

        return {'status': hyperopt.STATUS_OK, 'loss': np.power(validation_data.Demand.to_numpy() - fcast.to_numpy(), 2).mean()}

  search_space = {
      'p': scope.int(hyperopt.hp.quniform('p', 0, 4, 1)),
      'd': scope.int(hyperopt.hp.quniform('d', 0, 2, 1)),
      'q': scope.int(hyperopt.hp.quniform('q', 0, 4, 1)) 
    }

  rstate = np.random.default_rng(123) # just for reproducibility of this notebook

  best_hparams = fmin(evaluate_model, search_space, algo=tpe.suggest, max_evals=10)

  # Training
  model_final = SARIMAX(
    train_data["Demand"], 
    exog=train_data[exo_fields], 
    order=tuple(best_hparams.values()), 
    seasonal_order=(0, 0, 0, 0), # assume no seasonality in our example
    initialization_method="estimated",
    enforce_stationarity = False,
     enforce_invertibility = False
  )
  fitted_model_final = model_final.fit(disp=False, method='nm')

  # Validation
  fcast = fitted_model_final.predict(
    start=complete_ts.index.min(), 
    end=complete_ts.index.max(), 
    exog=validation_data[exo_fields]
  )

  return_series = complete_ts[['Product', 'SKU' , 'Demand']].assign(Demand_Fitted = fcast)
  
  return(return_series)

# COMMAND ----------

#Test
#import random
#random_sku = enriched_df.select('SKU').collect()[ random.randint(0, enriched_df.count()) ][ 0 ]
#sku_pdf = enriched_df.filter(F.col('SKU') == random_sku).toPandas()
#build_tune_and_score_model(sku_pdf)

# COMMAND ----------

tuning_schema = StructType(
  [
    StructField('Product', StringType()),
    StructField('SKU', StringType()),
    StructField('Demand', FloatType()),
    StructField('Demand_Fitted', FloatType())
  ]
)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Run distributed processing: `groupBy("SKU")` + `applyInPandas(...)`

# COMMAND ----------

forecast_df = (
  enriched_df
  .groupBy("Product", "SKU") 
  .applyInPandas(build_tune_and_score_model, schema=tuning_schema)
)
display(forecast_df)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Save to Delta

# COMMAND ----------

# Write the data 
forecast_df.write \
.mode("overwrite") \
.format("delta") \
.save('/FileStore/tables/demand_forecasting_solution_accelerator/forecast_df_delta/')

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS demand_db.part_level_demand_with_forecasts;
# MAGIC CREATE TABLE demand_db.part_level_demand_with_forecasts USING DELTA LOCATION '/FileStore/tables/demand_forecasting_solution_accelerator/forecast_df_delta/'

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM demand_db.part_level_demand_with_forecasts
