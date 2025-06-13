# Databricks notebook source
# MAGIC %md
# MAGIC # Fine Grained Demand Forecasting

# COMMAND ----------

# MAGIC %md
# MAGIC *Prerequisite: Make sure to run 01_Introduction_And_Setup before running this notebook.*
# MAGIC
# MAGIC In this notebook we use AutoML, or Automated Machine Learning. It simplifies the machine learning process by automatically identifying the best algorithms and hyperparameter configurations for specific datasets. Its main purpose is to make machine learning accessible to users with varying levels of expertise, allowing them to leverage the power of machine learning without extensive programming skills. 
# MAGIC
# MAGIC Key highlights for this notebook:
# MAGIC - AutoML

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
from pyspark.sql.functions import col

import databricks.automl
import logging
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Build a model
# MAGIC *while levaraging Databricks' AutoML capabilities*

# COMMAND ----------

# MAGIC %md 
# MAGIC Read in data

# COMMAND ----------

demand_df = spark.read.table(f"{catalogName}.{dbName}.part_level_demand") 
demand_df = demand_df.cache() # just for this example notebook

# COMMAND ----------

# MAGIC %md
# MAGIC > **⚠️ Next cell is only relevant for demos, you can comment it out to run training for the whole dataset, whenever you want. It takes around 10 minutes**

# COMMAND ----------

demand_df = demand_df.filter(demand_df.SKU == demand_df.first().SKU).cache()

# COMMAND ----------

display(demand_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Try the AutoML Forecast Function to find the best model automatically

# COMMAND ----------

# MAGIC %md
# MAGIC The following command starts an AutoML run. You must provide the column that the model should predict in the `target_col` argument and the time column. When the run completes, you can follow the link to the best trial notebook to examine the training code.
# MAGIC
# MAGIC This example also specifies:
# MAGIC
# MAGIC `horizon=40` to specify that AutoML should forecast 40 weeks into the future.  
# MAGIC `frequency="w"` to specify that a forecast should be provided for each week.  
# MAGIC `primary_metric="mse"` to specify the metric to optimize for during training.
# MAGIC
# MAGIC [AutoML Python API Reference](https://docs.databricks.com/aws/en/machine-learning/automl/automl-api-reference)
# MAGIC
# MAGIC **ℹ️ Takes around 8 minutes to run (full dataset)**
# MAGIC
# MAGIC **⚠️ Does currently not run on Serverless [06.05.2025]**

# COMMAND ----------

# Disable informational messages from fbprophet
logging.getLogger("py4j").setLevel(logging.WARNING)
 
# Note: If you are running Databricks Runtime for Machine Learning 10.4 or below, use this line instead:
# summary = databricks.automl.forecast(df, target_col="cases", time_col="date", horizon=30, frequency="d",  primary_metric="mdape")
 
summary = databricks.automl.forecast(demand_df, target_col="Demand", time_col="Date", identity_col="SKU", horizon=40, frequency="W", primary_metric="mse", output_database=dbName, exclude_frameworks=['prophet'], timeout_minutes=10)


# COMMAND ----------

# MAGIC %md
# MAGIC ##Load predictions from the best model
# MAGIC In Databricks Runtime for Machine Learning 10.5 or above, if output_database is provided, AutoML saves the predictions from the best model.
# MAGIC
# MAGIC

# COMMAND ----------

# Load the saved predictions.
forecast_pd = spark.table(summary.output_table_name)
display(forecast_pd)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Load the model with MLflow
# MAGIC MLflow allows you to easily import models back into Python by using the AutoML trial_id .

# COMMAND ----------

import mlflow.pyfunc
from mlflow.tracking import MlflowClient
 
run_id = MlflowClient()
trial_id = summary.best_trial.mlflow_run_id
 
model_uri = "runs:/{run_id}/model".format(run_id=trial_id)
pyfunc_model = mlflow.pyfunc.load_model(model_uri)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Use the model for forecasting
# MAGIC Call the predict_timeseries model method to generate forecasts.
# MAGIC In Databricks Runtime for Machine Learning 10.5 or above, you can set include_history=False to get the predicted data only.

# COMMAND ----------

forecasts = spark.createDataFrame(pyfunc_model._model_impl.python_model.predict_timeseries())
display(forecasts)

# COMMAND ----------

result_df = demand_df.join(forecasts.select(forecasts['ds'].cast('date').alias('Date'), col('yhat').alias('Demand_Fitted'),'SKU'), on=['Date', 'SKU'], how='right')
display(result_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Plot the forecasted points
# MAGIC In the plot below, the thick black line shows the time series dataset, and the blue line is the forecast created by the model.

# COMMAND ----------

df_res = result_df.toPandas()
fig, axes = plt.subplots(nrows=len(df_res['SKU'].unique()), ncols=1, facecolor='w', figsize=(10, 6 * len(df_res['SKU'].unique())))

for i, sku in enumerate(df_res['SKU'].unique()):
    ax = axes[i] if len(df_res['SKU'].unique()) > 1 else axes
    sku_data = df_res[df_res['SKU'] == sku]
    ax.plot(sku_data['Date'], sku_data['Demand'], 'k.', label=f'Observed data points - {sku}')
    ax.plot(sku_data['Date'], sku_data['Demand_Fitted'], ls='-', c='#0072B2', label=f'Forecasts - {sku}')
    ax.legend()
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save to Delta

# COMMAND ----------

result_df.write.mode("overwrite").saveAsTable(f"{catalogName}.{dbName}.part_level_demand_with_forecasts") 

# COMMAND ----------

display(spark.sql(f"SELECT * FROM {catalogName}.{dbName}.part_level_demand_with_forecasts"))
