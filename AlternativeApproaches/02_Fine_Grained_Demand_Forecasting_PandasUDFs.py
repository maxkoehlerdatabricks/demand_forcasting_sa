# Databricks notebook source
# MAGIC %md
# MAGIC # Fine Grained Demand Forecasting

# COMMAND ----------

# MAGIC %md
# MAGIC *Prerequisite: Make sure to run 01_Introduction_And_Setup before running this notebook.*
# MAGIC
# MAGIC In this notebook we first find an appropriate time series model and then apply that very same approach to train multiple models in parallel with great speed and cost-effectiveness.  
# MAGIC
# MAGIC Key highlights for this notebook:
# MAGIC - Use Databricks' collaborative and interactive notebook environment to find an appropriate time series mdoel
# MAGIC - Pandas UDFs (user-defined functions) can take your single-node data science code, and distribute it across different keys (e.g. SKU)  
# MAGIC - Use Ray to perform hyperparameter tuning from within a Pandas UDF  

# COMMAND ----------

dbutils.widgets.dropdown('reset_all_data', 'false', ['true', 'false'], 'Reset all data')
dbutils.widgets.text('catalogName',  'maxkoehler_demos' , 'Catalog Name')
dbutils.widgets.text('dbName',  'demand_db' , 'Database Name')

# COMMAND ----------

catalogName = dbutils.widgets.get('catalogName')
dbName = dbutils.widgets.get('dbName')
reset_all_data = dbutils.widgets.get('reset_all_data') == 'true'

# COMMAND ----------

# MAGIC %run ./_resources/00-setup $reset_all_data=false $catalogName=$catalogName $dbName=$dbName 

# COMMAND ----------

import os
import datetime as dt
from random import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as md

from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from statsmodels.tsa.statespace.sarimax import SARIMAX

import mlflow
import ray
from ray import tune
from ray.tune.search.hyperopt import HyperOptSearch

mlflow.autolog(disable=True)



from statsmodels.tsa.api import Holt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_percentage_error

import pyspark.sql.functions as f
from pyspark.sql.types import *

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Build a model
# MAGIC *while levaraging Databricks' collaborative and interactive environment*

# COMMAND ----------

# MAGIC %md 
# MAGIC Read in data

# COMMAND ----------

demand_df = spark.read.table(f"part_level_demand")
demand_df = demand_df.cache() # just for this example notebook

# COMMAND ----------

# MAGIC %md
# MAGIC ### Examine an example: Extract a single time series and convert to pandas dataframe

# COMMAND ----------

example_sku = demand_df.select("SKU").orderBy("SKU").limit(1).collect()[0].SKU
print("example_sku:", example_sku)
pdf = demand_df.filter(f.col("SKU") == example_sku).toPandas()

# Create single series 
series_df = pd.Series(pdf['Demand'].values, index=pdf['Date'])
series_df = series_df.asfreq(freq='W-MON')

display(pdf)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Determine the forecast horizon

# COMMAND ----------

forecast_horizon = 40
is_history = [True] * (len(series_df) - forecast_horizon) + [False] * forecast_horizon
train = series_df.iloc[is_history]
score = series_df.iloc[~np.array(is_history)]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Derive exogenous variables

# COMMAND ----------

covid_breakpoint = dt.date(year=2020, month=3, day=1)
exo_df = pdf.assign(Week = pd.DatetimeIndex(pdf["Date"]).isocalendar().week.tolist()) 

exo_df = exo_df \
  .assign(covid = np.where(pdf["Date"] >= np.datetime64(covid_breakpoint), 1, 0).tolist()) \
  .assign(christmas = np.where((exo_df["Week"] >= 51) & (exo_df["Week"] <= 52) , 1, 0).tolist()) \
  .assign(new_year = np.where((exo_df["Week"] >= 1) & (exo_df["Week"] <= 4)  , 1, 0).tolist()) \
  .set_index('Date')

exo_df = exo_df[["covid", "christmas", "new_year" ]]
exo_df = exo_df.asfreq(freq='W-MON')
print(exo_df)
train_exo = exo_df.iloc[is_history]  
score_exo = exo_df.iloc[~np.array(is_history)]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Try the Holt’s Winters Seasonal Method

# COMMAND ----------

# MAGIC %md
# MAGIC The Holt's Winters Seasonal method models seasonal and trend components. Diffreent versions of it serve as a good first try. We observe a good fit to adopt to the time series and its irregular components in the training period but not in the forecasting horizon. This model does not do a good job for forecasting the Christmas effect or the recovery period after the pandemic started. 

# COMMAND ----------

fit1 = ExponentialSmoothing(
    train,
    seasonal_periods=3,
    trend="add",
    seasonal="add",
    use_boxcox=True,
    initialization_method="estimated",
).fit(method="ls")

fcast1 = fit1.forecast(forecast_horizon).rename("Additive trend and additive seasonal")

fit2 = ExponentialSmoothing(
    train,
    seasonal_periods=4,
    trend="add",
    seasonal="mul",
    use_boxcox=True,
    initialization_method="estimated",
).fit(method="ls")

fcast2 = fit2.forecast(forecast_horizon).rename("Additive trend and multiplicative seasonal")

fit3 = ExponentialSmoothing(
    train,
    seasonal_periods=4,
    trend="add",
    seasonal="add",
    damped_trend=True,
    use_boxcox=True,
    initialization_method="estimated",
).fit(method="ls")

fcast3 = fit3.forecast(forecast_horizon).rename("Additive damped trend and additive seasonal")

fit4 = ExponentialSmoothing(
    train,
    seasonal_periods=4,
    trend="add",
    seasonal="mul",
    damped_trend=True,
    use_boxcox=True,
    initialization_method="estimated",
).fit(method="ls")


fcast4 = fit4.forecast(forecast_horizon).rename("Additive damped trend and multiplicative seasonal")

plt.figure(figsize=(12, 8))
(line0,) =  plt.plot(series_df, marker="o", color="black")
plt.plot(fit1.fittedvalues, color="blue")
(line1,) = plt.plot(fcast1, marker="o", color="blue")
plt.plot(fit2.fittedvalues, color="red")
(line2,) = plt.plot(fcast2, marker="o", color="red")
plt.plot(fit3.fittedvalues, color="green")
(line3,) = plt.plot(fcast3, marker="o", color="green")
plt.plot(fit4.fittedvalues, color="orange")
(line4,) = plt.plot(fcast4, marker="o", color="orange")
plt.axvline(x = min(score.index.values), color = 'red', label = 'axvline - full height')
plt.legend([line0, line1, line2, line3, line4], ["Actuals", fcast1.name, fcast2.name, fcast3.name, fcast4.name])
plt.xlabel("Time")
plt.ylabel("Demand")
plt.title("Holts Winters Seasonal Method")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Try the SARIMAX method

# COMMAND ----------

# MAGIC %md
# MAGIC A SARIMAX model allows to incorporate explanatory variables. From a business point of view, this helps to incorporate business knowledge about demand driving events. This could not only be a christmas effect, but also promotion actions. We observe that the model does a poor job when not taking advantage of the business knowledge. However, if incorporating exogenous variables, the Christmas effect and the after-pandemic trend can fit well in the forecasting horizon.

# COMMAND ----------

# MAGIC %md
# MAGIC #### First model

# COMMAND ----------

fit1 = SARIMAX(train, order=(1, 2, 1), seasonal_order=(0, 0, 0, 0), initialization_method="estimated").fit(warn_convergence = False)
fcast1 = fit1.predict(start = min(train.index), end = max(score_exo.index)).rename("Without exogenous variables")

fit2 = SARIMAX(train, exog=train_exo, order=(1, 2, 1), seasonal_order=(0, 0, 0, 0), initialization_method="estimated").fit(warn_convergence = False)
fcast2 = fit2.predict(start = min(train.index), end = max(score_exo.index), exog = score_exo).rename("With exogenous variables")

# COMMAND ----------

plt.figure(figsize=(18, 6))
plt.plot(series_df, marker="o", color="black")
plt.plot(fcast1[10:], color="blue")
(line1,) = plt.plot(fcast1[10:], marker="o", color="blue")
plt.plot(fcast2[10:], color="green")
(line2,) = plt.plot(fcast2[10:], marker="o", color="green")

plt.axvline(x = min(score.index.values), color = 'red', label = 'axvline - full height')
plt.legend([line0, line1, line2], ["Actuals", fcast1.name, fcast2.name])
plt.xlabel("Time")
plt.ylabel("Demand")
plt.title("SARIMAX")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Taking advantage of MLFlow and Hyperopt to find optimal parameters in the SARIMAX model

# COMMAND ----------

# Define the search space
space = {
  'p': tune.randint(0, 5),
  'd': tune.randint(0, 3),
  'q': tune.randint(0, 5)
}

# COMMAND ----------

def evaluate_model(config):
    try:
        order_parameters = (config["p"], config["d"], config["q"])

        # For simplicity in this example, assume no seasonality
        model1 = SARIMAX(train, exog=train_exo, order=order_parameters, seasonal_order=(0, 0, 0, 0))
        fit1 = model1.fit(disp=False)
        fcast1 = fit1.predict(start=min(score_exo.index), end=max(score_exo.index), exog=score_exo)
        mse = np.power(score.to_numpy() - fcast1.to_numpy(), 2).mean()
        return {"mse": mse}
    except Exception as e:
        print(f"Error in evaluate_model: {e}")
        return {"mse": float('inf')}

# Initialize Ray
ray.init()

# Start the hyperparameter tuning
with mlflow.start_run(run_name=f'maxkoehler_test_sa'):
    analysis = tune.run(
        evaluate_model,
        config=space,
        search_alg=HyperOptSearch(
            metric="mse",
            mode="min"
        ),
        num_samples=10,
        verbose=1
    )

# Shutdown Ray
ray.shutdown()

# COMMAND ----------

best_config = analysis.get_best_config(metric="mse", mode="min")
print("Best hyperparameters: ", best_config)

# COMMAND ----------

# MAGIC %md
# MAGIC In the next step we scale this procedure to thousands of models.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Train thousands of models at scale
# MAGIC *while still using your preferred libraries and approaches*

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
# MAGIC Let's have a look at our core dataset, this time with exogenous variables. This time, data for all SKUs is logically unified within a Spark DataFrame, allowing large-scale distributed processing.

# COMMAND ----------

enriched_df = (
  demand_df
    .groupBy("Product")
    .applyInPandas(add_exo_variables, enriched_schema)
)
display(enriched_df)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ### Solution: High-Level Overview
# MAGIC
# MAGIC Benefits:
# MAGIC - Pure Python & Pandas: easy to develop, test
# MAGIC - Continue using your favorite libraries
# MAGIC - Simply assume you're working with a Pandas DataFrame for a single SKU
# MAGIC
# MAGIC <img src="https://github.com/PawaritL/data-ai-world-tour-dsml-jan-2022/blob/main/pandas-udf-workflow.png?raw=true" width=40%>

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Build, tune and score a model per each SKU with Pandas UDFs

# COMMAND ----------

def build_tune_and_score_model(sku_pdf: pd.DataFrame) -> pd.DataFrame:
  """
  This function trains, tunes and scores a model for each SKU and can be distributed as a Pandas UDF
  """
  # Always ensure proper ordering and indexing by Date
  sku_pdf.sort_values("Date", inplace=True)
  complete_ts = sku_pdf.set_index("Date").asfreq(freq="W-MON")
  
  print(complete_ts)
  
  # Since we'll group the large Spark DataFrame by (Product, SKU)
  PRODUCT = sku_pdf["Product"].iloc[0]
  SKU = sku_pdf["SKU"].iloc[0]
  train_data, validation_data = split_train_score_data(complete_ts)
  exo_fields = ["covid", "christmas", "new_year"]

  # Evaluate model on the training data set
  def evaluate_model(config):
    try:
      order_parameters = (config["p"], config["d"], config["q"])

      # SARIMAX requires a tuple of Python integers
      order_hparams = tuple([int(config[k]) for k in ("p", "d", "q")])

      # Training
      model = SARIMAX(
        train_data["Demand"], 
        exog=train_data[exo_fields], 
        order=order_hparams, 
        seasonal_order=(0, 0, 0, 0), # assume no seasonality in our example
        initialization_method="estimated",
        enforce_stationarity=False,
        enforce_invertibility=False
      )
      fitted_model = model.fit(disp=False, method='nm')

      # Validation
      fcast = fitted_model.predict(
        start=validation_data.index.min(), 
        end=validation_data.index.max(), 
        exog=validation_data[exo_fields]
      )

      mse = np.power(validation_data["Demand"].to_numpy() - fcast.to_numpy(), 2).mean()
      return {"mse": mse}
    
    except Exception as e:
      print(f"Error in evaluate_model: {e}")
      return {"mse": float('inf')}
       

  #best_config = analysis.get_best_config(metric="mse", mode="min")
  best_config = {'p': 2, 'd': 2, 'q': 1}

  # Training
  model_final = SARIMAX(
    train_data["Demand"], 
    exog=train_data[exo_fields], 
    order=tuple(best_config.values()), 
    seasonal_order=(0, 0, 0, 0), # assume no seasonality in our example
    initialization_method="estimated",
    enforce_stationarity=False,
    enforce_invertibility=False
  )
  fitted_model_final = model_final.fit(disp=False, method='nm')

  # Validation
  fcast = fitted_model_final.predict(
    start=complete_ts.index.min(), 
    end=complete_ts.index.max(), 
    exog=validation_data[exo_fields]
  )

  forecast_series = complete_ts[['Product', 'SKU', 'Demand']].assign(Date=complete_ts.index.values).assign(Demand_Fitted=fcast)
  forecast_series = forecast_series[['Product', 'SKU', 'Date', 'Demand', 'Demand_Fitted']]
  
  return forecast_series

# COMMAND ----------

tuning_schema = StructType(
  [
    StructField('Product', StringType()),
    StructField('SKU', StringType()),
    StructField('Date', DateType()),
    StructField('Demand', FloatType()),
    StructField('Demand_Fitted', FloatType())
  ]
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Run distributed processing: `groupBy("SKU")` + `applyInPandas(...)`

# COMMAND ----------

forecast_df = (
  enriched_df
  .groupBy("Product", "SKU") 
  .applyInPandas(build_tune_and_score_model, schema=tuning_schema)
)
display(forecast_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save to Delta

# COMMAND ----------

forecast_df.write.mode("overwrite").saveAsTable("part_level_demand_with_forecasts")

# COMMAND ----------

display(spark.sql(f"SELECT * FROM part_level_demand_with_forecasts"))
