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
# MAGIC - MLflow can track and log all of your parameters, metrics, and artifacts - which can be loaded for later use

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Getting to Production: Industry Pain Points
# MAGIC 
# MAGIC <img src="https://github.com/PawaritL/data-ai-world-tour-dsml-jan-2022/blob/main/pain-points.png?raw=true" width=80%>

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
# MAGIC #### Build and tune a model per each SKU with Pandas UDFs

# COMMAND ----------

def build_and_tune_model(sku_pdf: pd.DataFrame) -> pd.DataFrame:
  """
  This function trains and tunes a model for each SKU and can be distributed as a Pandas UDF
  """
  # Always ensure proper ordering and indexing by Date
  sku_pdf.sort_values("Date", inplace=True)
  complete_ts = sku_pdf.set_index("Date").asfreq(freq="W-MON")
  
  # Since we'll group the large Spark DataFrame by (Product, SKU)
  PRODUCT = sku_pdf["Product"].iloc[0]
  SKU = sku_pdf["SKU"].iloc[0]
  train_data, validation_data = split_train_score_data(complete_ts)
  
  def _score(hparams, final_fit=False):
    in_sample_data = complete_ts if final_fit else train_data
    score_data = complete_ts if final_fit else validation_data
    exo_fields = ["covid", "christmas", "new_year"]
    # SARIMAX's predict() only accepts exog variables from new (out-of-sample) dates 
    exog = score_data[exo_fields].loc[score_data.index > in_sample_data.index.max()]
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
    fcast = fitted_model.predict(
      start=score_data.index.min(), 
      end=score_data.index.max(), 
      exog=exog
    )

    loss = np.power(score_data["Demand"].to_numpy() - fcast.to_numpy(), 2).mean()
    mape = mean_absolute_percentage_error(score_data["Demand"].to_numpy(), fcast.to_numpy())
    
    if final_fit: # for returning final model + key metrics
      return loss, fitted_model, mape
    else: # for standard hyperparameter tuning
      return loss
  
  # Iterate over search space for ARIMA parameters
  search_space = {
    'p': scope.int(hyperopt.hp.quniform('p', 0, 4, 1)),
    'd': scope.int(hyperopt.hp.quniform('d', 0, 2, 1)),
    'q': scope.int(hyperopt.hp.quniform('q', 0, 4, 1)) 
  }
  rstate = np.random.RandomState(123) # just for reproducibility of this notebook
  
  best_hparams = fmin(_score, search_space, algo=tpe.suggest, max_evals=10)
  # Perform final fit and evaluation (after getting best hyperparameters)
  loss, fitted_model, mape = _score(best_hparams, final_fit=True)
  
  return pd.DataFrame(
    {
      'Product':[PRODUCT], 
      'SKU':[SKU], 
      'mean_absolute_percentage_error': [mape], 
      'mean_squared_error': [loss],
      'best_hparams':[json.dumps(best_hparams) if loss != np.inf else "FAILED"], 
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
# MAGIC ### Run distributed processing: `groupBy("SKU")` + `applyInPandas(...)`

# COMMAND ----------

tuned_df = (
  enriched_df
  .groupBy("Product", "SKU") 
  .applyInPandas(build_and_tune_model, schema=tuning_schema)
)
tuned_df = tuned_df.cache() # reuse these results later in our workflow
display(tuned_df)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Log and organize details at the Product level

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Tracking parameters, metrics, and model artifacts in an organized way
# MAGIC 
# MAGIC While we could've also returned forecast results as the output of the Pandas UDF, it's often be beneficial to break down our workflow into a few steps:
# MAGIC 1. Model training and tuning by SKU
# MAGIC 2. Parameter, metric, and model logging into MLflow - *organizing runs by Product* <br> (so that results and artifacts are a bit more well-organized and easier to retrieve)
# MAGIC 3. Loading saved models from MLflow for inference (i.e. running your forecasts)
# MAGIC 
# MAGIC Modularizing your workflow can be extremely helpful for ensuring reproducibility and debuggability!  
# MAGIC 
# MAGIC #### Approach: Nested MLflow Runs
# MAGIC <img src="https://miro.medium.com/max/1400/1*i0KMm30k8cPbu-eTVBmUZQ.gif" width=69%>  <br>  
# MAGIC *GIF by: [Patryk Oleniuk](https://towardsdatascience.com/5-tips-for-mlflow-experiment-tracking-c70ae117b03f)*

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
    # SARIMAX's predict() only accepts exog variables from new (out-of-sample) dates 
    out_of_sample_exog = ts[exo_fields].loc[ts.index > self.last_known_exog_date]
    fcast = self.fitted_model.predict(
      start=ts.index.min(), 
      end=ts.index.max(), 
      exog=out_of_sample_exog
    )
    return fcast

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### MLflow Logging Pandas UDF

# COMMAND ----------

def log_to_mlflow(product_pdf: pd.DataFrame) -> pd.DataFrame:
  """
  In case there are many thousands of SKUs, it may help to introduce a hierarchy (nesting) of runs.
  We can have a parent run at the Product level, then a child run per each SKU
  """
  PRODUCT = product_pdf["Product"].iloc[0]
  SKU_LIST = list(product_pdf["SKU"].unique())
  
  with mlflow.start_run(run_name=PRODUCT) as parent_run: # nest MLflow runs for easier organization
    mlflow.log_param("Product", PRODUCT)
    mlflow.log_param("SKUs", SKU_LIST)
    mlflow.log_metric("max_mape", product_pdf["mean_absolute_percentage_error"].max())
    mlflow.log_metric("mean_mape", product_pdf["mean_absolute_percentage_error"].mean())
    
    for idx, sku_row in product_pdf.iterrows():
      SKU = sku_row["SKU"]
      with mlflow.start_run(run_name=SKU, nested=True) as child_run:
        mlflow.log_param("Product", PRODUCT)
        mlflow.log_param("SKU", SKU)
        mlflow.log_param("best_hparams", sku_row["best_hparams"])
        mlflow.log_metric("mape", sku_row["mean_absolute_percentage_error"])
        # mlflow.log_artifact(...) # you can even log artifacts e.g. plots/charts
      
        best_hparams = json.loads(sku_row["best_hparams"])
        fitted_model = pickle.loads(sku_row["model_binary"])
        sku_model = SKUModelWrapper(fitted_model)
        # Log each SKU model
        mlflow.pyfunc.log_model(
          f"{PRODUCT.replace(' ', '_')}_{SKU}_SARIMAX_model", 
          python_model=sku_model
        )
      
  return pd.DataFrame({"Product": [PRODUCT], "Status": ["FINISHED"], "SKUs": [SKU_LIST]})

# COMMAND ----------

unique_products = tuned_df.select("Product").dropDuplicates().toPandas()
n_unique_products = len(unique_products)

spark.conf.set("spark.sql.shuffle.partitions", n_unique_products)

logging_return_schema = StructType(
  [
    StructField("Product", StringType()),
    StructField("Status", StringType()),
    StructField("SKUs", ArrayType(StringType()))
  ]
)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Run logging, then view training/tuning details in MLflow Experiments UI

# COMMAND ----------

logged_df = (
  tuned_df
  .groupBy("Product")
  .applyInPandas(log_to_mlflow, schema=logging_return_schema)
)
display(logged_df)

# COMMAND ----------

import mlflow

example_sku = "CAM_0X6CLF"

# Alternatively, please visit the MLflow experiment tracking UI for an example of model loading
# https://docs.databricks.com/applications/mlflow/models.html#automatically-generated-code-snippets-in-the-mlflow-ui
latest_search_result = mlflow.search_runs(filter_string=f"params.SKU='{example_sku}'").iloc[0]
logged_model_metadata = json.loads(
  latest_search_result["tags.mlflow.log-model.history"]
)
logged_model_metadata

example_run_id = logged_model_metadata[0]["run_id"]
example_artifact_path = logged_model_metadata[0]["artifact_path"]

logged_model = f"runs:/{example_run_id}/{example_artifact_path}"

# COMMAND ----------

import pandas as pd

example_sku_pdf = enriched_df.filter(F.col("SKU") == example_sku).toPandas()

incomplete_lag_points = 10 # as we can't compute lag values on the earliest data points
fcast_weeks = 40
in_sample_pdf = example_sku_pdf.iloc[incomplete_lag_points:].copy()

out_of_sample_pdf = pd.DataFrame()
out_of_sample_pdf["Date"] = [in_sample_pdf["Date"].max() + pd.Timedelta(days=7*(i+1)) for i in range(fcast_weeks)]
out_of_sample_pdf["Product"] = in_sample_pdf["Product"].iloc[0]
out_of_sample_pdf["SKU"] = example_sku
out_of_sample_pdf["Demand"] = np.nan
out_of_sample_pdf = add_exo_variables(out_of_sample_pdf)

example_inference_pdf = pd.concat([in_sample_pdf, out_of_sample_pdf], axis=0)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify results after loading saved model from MLflow

# COMMAND ----------

import mlflow
# Load model as a PyFuncModel
# You can also load a saved model as a Spark UDF for large-scale batch inference
# https://docs.databricks.com/applications/mlflow/model-example.html
loaded_model = mlflow.pyfunc.load_model(logged_model)
example_inference_pdf["prediction"] = loaded_model.predict(example_inference_pdf).values

# COMMAND ----------

import plotly.express as px
    
fig = px.line(example_inference_pdf, x="Date", y="Demand", title=f"Part-Level Forecast: SKU={example_sku}")
fig.update_traces(name="True Demand", showlegend=True)

in_sample_predictions_pdf = example_inference_pdf.loc[example_inference_pdf["Date"] <= in_sample_pdf["Date"].max()]
out_of_sample_forecast_pdf = example_inference_pdf.loc[example_inference_pdf["Date"] > in_sample_pdf["Date"].max()]

fig.add_scatter(
  x=in_sample_predictions_pdf["Date"], y=in_sample_predictions_pdf["prediction"], 
  mode="markers", 
  marker=dict(
    size=5, 
    color="LightSeaGreen",
    opacity=0.69
  ), 
  name="In-Sample Predictions"
)

fig.add_scatter(
  x=out_of_sample_forecast_pdf["Date"], y=out_of_sample_forecast_pdf["prediction"], 
  mode="markers", 
  marker=dict(
    size=5, 
    color="Orange",
    opacity=0.69
  ), 
  name="Out-of-Sample Forecast"
)

fig.update_layout(yaxis = dict(range=[4000, 18000]))
fig.add_vline(x=in_sample_pdf["Date"].max(), line_width=3, line_dash="dash")
fig.show()

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ## Key Takeaways
# MAGIC 
# MAGIC <br>  
# MAGIC 
# MAGIC 1. **Pandas UDFs on Databricks** enable flexible distributed computing, while still using your favorite single-node libraries  
# MAGIC 2. <a>**MLflow**</a> helps productionize your model development workflow to ensure traceability and reproducibility  
# MAGIC 3. ***No MLOps without DataOps:*** <br> <a>**Delta Lake**</a> allows you to save data/forecasts to your data lake with proper data management <br> (e.g. data versioning, time travel, fine-grained updates/deletes)
# MAGIC 
# MAGIC <img src="https://github.com/PawaritL/data-ai-world-tour-dsml-jan-2022/blob/main/pandas-udf-workflow.png?raw=true" width=38%>
