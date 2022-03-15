# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Train thousands of models at scale
# MAGIC *while still using your preferred libraries and approaches*
# MAGIC 
# MAGIC **Prerequisite: Make sure to run the data simulation notebook 01_Simulate_Data before running this notebook.**  
# MAGIC 
# MAGIC While the previous notebook *(02_Explore_Data)* demonstrated the benefits of Databricks' collaborative and interactive environment,  
# MAGIC in this final part, we can apply that very same approach to train multiple models in parallel with great speed and cost-effectiveness.  
# MAGIC 
# MAGIC Key highlights for this notebook:
# MAGIC - Pandas UDFs (user-defined functions) can take your single-node data science code, and distribute it across different keys (e.g. SKU)  
# MAGIC - Hyperopt can also perform hyperparameter tuning from within a Pandas UDF  
# MAGIC - MLflow can track and log all of your parameters, metrics, and artifacts - which can be loaded for later use

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

# DBTITLE 1,Modularize single-node logic from before into Python functions
FORECAST_HORIZON = 40

def add_exo_variables(pdf: pd.DataFrame) -> pd.DataFrame:
  
  midnight = dt.datetime.min.time()
  timestamp = pdf["Date"].apply(lambda x: dt.datetime.combine(x, midnight))
  calendar_week = timestamp.dt.isocalendar().week
  
  # define flexible, custom logic for exogenous variables
  covid_breakpoint = dt.datetime(year=2020, month=3, day=1)
  enriched_df = (
    pdf
      .assign(covid = (timestamp >= covid_breakpoint).astype(int))
      .assign(christmas = ((calendar_week >= 51) & (calendar_week <= 52)).astype(int))
      .assign(new_year = ((calendar_week >= 1) & (calendar_week <= 4)).astype(int))
  )
  return enriched_df[["Date", "Product", "SKU", "Demand", "covid", "christmas", "new_year"]]

enriched_schema = StructType(
  [
    StructField('Date', DateType()),
    StructField('Product', StringType()),
    StructField('SKU', StringType()),
    StructField('Demand', FloatType()),
    StructField('covid', IntegerType()),
    StructField('christmas', IntegerType()),
    StructField('new_year', IntegerType()),
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

# COMMAND ----------

data_path = '/FileStore/tables/demand_forecasting_solution_accellerator/demand_df_delta/'
demand_df = (
  spark
    .read
    .format("delta")
    .load(data_path)
)
demand_df = demand_df.cache()

enriched_df = (
  demand_df
    .groupBy("Product")
    .applyInPandas(add_exo_variables, enriched_schema)
)
display(enriched_df)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Pandas function to build and tune a model per each SKU

# COMMAND ----------

def build_and_tune_model(sku_pdf: pd.DataFrame) -> pd.DataFrame:
  """
  This function trains and tunes a model for each SKU and can be distributed as a Pandas UDF
  """
  # Always ensure proper ordering and indexing by Date
  sku_pdf.sort_values("Date", inplace=True)
  sku_ts = sku_pdf.set_index("Date").asfreq(freq="W-MON")
  
  # Since we'll group the large Spark DataFrame by (Product, SKU)
  PRODUCT = sku_pdf["Product"].iloc[0]
  SKU = sku_pdf["SKU"].iloc[0]
  
  non_holdout_data, holdout_data = split_train_score_data(sku_ts)
  train_data, validation_data = split_train_score_data(non_holdout_data)
  
  # Search space for ARIMA parameters
  search_space = {
    'p': scope.int(hyperopt.hp.quniform('p', 0, 4, 1)),
    'd': scope.int(hyperopt.hp.quniform('d', 0, 4, 1)),
    'q': scope.int(hyperopt.hp.quniform('q', 0, 4, 1)) 
  }
  
  def _score(hparams, final_fit=False):
    in_sample_data = non_holdout_data if final_fit else train_data
    score_data = holdout_data if final_fit else validation_data
    exo_fields = ["covid", "christmas", "new_year"]
    
    # SARIMAX requires a tuple of Python integers
    order_hparams = tuple([int(hparams[k]) for k in ("p", "d", "q")])
    
    model = SARIMAX(
      in_sample_data["Demand"], 
      exog=in_sample_data[exo_fields], 
      order=order_hparams, 
      seasonal_order=(0, 0, 0, 0), # assume no seasonality in our example
      initialization_method="estimated"
    )
    fitted_model = model.fit(disp=False)
    fcast = fitted_model.predict(start=score_data.index.min(), end=score_data.index.max(), exog=score_data[exo_fields])
    
    loss = np.power(score_data["Demand"].to_numpy() - fcast.to_numpy(), 2).mean()
    mape = mean_absolute_percentage_error(score_data["Demand"].to_numpy(), fcast.to_numpy())
    
    if final_fit: # for returning final model + key metrics
      return loss, fitted_model, mape
    else: # for standard hyperparameter tuning
      return loss
    
  # Iterate over search space
  best_hparams = fmin(_score, search_space, algo=tpe.suggest, max_evals=10)
  # Perform final fit and evaluation (after getting best hyperparameters)
  loss, fitted_model, mape = _score(best_hparams, final_fit=True)
  
  return pd.DataFrame(
    {
      'Product':[PRODUCT], 
      'SKU':[SKU], 
      'mean_absolute_percentage_error': [mape], 
      'mean_squared_error': [loss],
      'best_hparams':[json.dumps(best_hparams)], 
      'model_binary': [pickle.dumps(fitted_model)]
    }
  )

# COMMAND ----------

demand_pdf = demand_df.toPandas()
unique_skus = demand_df.select("SKU").dropDuplicates().toPandas()
n_unique_skus = len(unique_skus)

spark.conf.set("spark.sql.shuffle.partitions", n_unique_skus)

tuning_schema = StructType(
  [
    StructField('Product', StringType()),
    StructField('SKU', StringType()),
    StructField('mean_absolute_percentage_error', FloatType()),
    StructField('mean_squared_error', FloatType()),
    StructField('best_hparams', StringType()),
    StructField('model_binary', BinaryType()),
  ]
)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Run Pandas UDF grouped at the SKU level

# COMMAND ----------

tuned_df = (
  enriched_df
  .groupBy("Product", "SKU")
  .applyInPandas(build_and_tune_model, schema=tuning_schema)
)

tuned_df = tuned_df.cache() # cache the tuning results, most likely will be re-using them!
display(tuned_df.select("Product", "SKU", "mean_absolute_percentage_error", "mean_squared_error", "best_hparams"))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Log models & information at the Product level (for easy organization)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Let's log the parameters, metrics, and model artifacts in an organized manne
# MAGIC 
# MAGIC Although we could've simply returned the forecast results as the output of the Pandas UDF above, it can often be beneficial to break down our workflow into a few steps:
# MAGIC 1. Model training and tuning by SKU
# MAGIC 2. Parameter, metric, and model logging into MLflow - *perhaps by Product* (so that results and artifacts are a bit more well-organized and easier to retrieve)
# MAGIC 3. Loading saved models from MLflow for inference (i.e. running your forecasts)
# MAGIC 
# MAGIC Modularizing your workflow can be extremely helpful for ensuring reproducibility and debuggability!

# COMMAND ----------

class SKUModelWrapper(mlflow.pyfunc.PythonModel):

  def __init__(self, fitted_model):
    self.fitted_model = fitted_model
    self.last_known_exog_date = fitted_model.mlefit.model.orig_exog.index.max()
    
  def load_context(self, context):
    # Optional: load artifacts if needed
    # https://www.mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.PythonModel.load_context
    pass

  def predict(self, context, data: pd.DataFrame):
    
    exo_fields = ["covid", "christmas", "new_year"]
    expected_fields = ["Date"] + exo_fields
    
    assert isinstance(data, pd.DataFrame), f"Expected pd.DataFrame, got: {type(data)}"
    assert all(x in data.columns for x in expected_fields)
    
    ts = data.set_index("Date").asfreq(freq="W-MON")
    # statsmodels implementation of SARIMAX requires new dates for exog. variables only
    out_of_sample_exog = ts[exo_fields].loc[ts.index > self.last_known_exog_date]
    fcast = self.fitted_model.predict(
      start=ts.index.min(), 
      end=ts.index.max(), 
      exog=out_of_sample_exog
    )
    return fcast

# COMMAND ----------

def log_to_mlflow(product_pdf: pd.DataFrame) -> pd.DataFrame:
  """
  In case there are many thousands of SKUs, it might be overwhelming to log/retrieve data to/from the MLflow server for every individual SKU.
  Instead we can group the SKU models/artifacts by e.g. Product to create a more organized hierarchy 
  """
  PRODUCT = product_pdf["Product"].iloc[0]
  
  # MLOps: parameter, metric, and artifact logging
  # we can use nested MLflow runs for easier organization
  with mlflow.start_run(run_name=PRODUCT) as parent_run:
    mlflow.log_param("Product", PRODUCT)
    
    for idx, sku_row in product_pdf.iterrows():
      SKU = sku_row["SKU"]
      with mlflow.start_run(run_name=SKU, nested=True) as child_run:
        mlflow.log_param("sku", SKU)
        mlflow.log_param("best_hparams", sku_row["best_hparams"])
        mlflow.log_metric("mean_absolute_percentage_error", sku_row["mean_absolute_percentage_error"])
        # you can even log artifacts such as plots/charts
        # mlflow.log_artifact(...)
      
        best_hparams = json.loads(sku_row["best_hparams"])
        fitted_model = pickle.loads(sku_row["model_binary"])
        sku_model = SKUModelWrapper(fitted_model)
        # Finally, log model - each product model contains underlying SARIMAX models for each SKU
        mlflow.pyfunc.log_model(
          f"{PRODUCT.replace(' ', '_')}_{SKU}_SARIMAX_model", 
          python_model=sku_model
        )
      
  return pd.DataFrame({"Product": [PRODUCT], "Status": ["FINISHED"]})

# COMMAND ----------

unique_products = tuned_df.select("Product").dropDuplicates().toPandas()
n_unique_products = len(unique_products)

spark.conf.set("spark.sql.shuffle.partitions", n_unique_products)

logging_return_schema = StructType(
  [
    StructField('Product', StringType()),
    StructField('Status', StringType()),
  ]
)

logged_df = (
  tuned_df
  .groupBy("Product")
  .applyInPandas(log_to_mlflow, schema=logging_return_schema)
)

display(logged_df)

# COMMAND ----------

# DBTITLE 1,Example: load saved model from MLflow
import mlflow
logged_model = 'runs:/b49d40caf9df47838aff0e3a1af1aafc/Short_Range_Lidar_SRR_1IDZ5T_SARIMAX_model'

# Load model as a PyFuncModel
# You can also load a saved model as a Spark UDF for large-scale batch inference
# https://docs.databricks.com/applications/mlflow/model-example.html
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd

example_sku = "SRR_1IDZ5T"
example_sku_pdf = enriched_df.filter(F.col("SKU") == example_sku).toPandas()
example_inference_pdf = example_sku_pdf.copy()[10:]
example_inference_pdf["prediction"] = loaded_model.predict(example_inference_pdf).values

# COMMAND ----------

# DBTITLE 1,Verify results after loading saved model from MLflow
import plotly.express as px
    
fig = px.line(example_inference_pdf, x="Date", y="Demand", title=f"Demand Forecast for SKU: {example_sku}")
fig.update_traces(name="True Demand", showlegend=True)
fig.add_scatter(
  x=example_inference_pdf["Date"], y=example_inference_pdf["prediction"], 
  mode="markers", 
  marker=dict(
    size=5, 
    color="LightSeaGreen"
  ), 
  name="Predicted"
)
fig.update_layout(yaxis = dict(range=[4000, 18000]))
fig.add_vline(x="2020-10-12", line_width=3, line_dash="dash")
fig.show()
