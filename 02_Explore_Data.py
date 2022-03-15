# Databricks notebook source
# MAGIC %md
# MAGIC # Data Exploration
# MAGIC This notebook explores the analysis data and trains first models on them. Make sure to run the data simulation notebook 01_Simulate_Data before running this notebook.

# COMMAND ----------

# MAGIC %md
# MAGIC # Packages and Parameters

# COMMAND ----------

import os
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
from hyperopt import hp, fmin, tpe, SparkTrials, STATUS_OK, space_eval
from hyperopt.pyll.base import scope
mlflow.autolog(disable=True)

import pyspark.sql.functions as F
from pyspark.sql.functions import col

# COMMAND ----------

forecast_horizon = 40
covid_breakpoint = datetime.date(year=2020, month=3, day=1)

# COMMAND ----------

# MAGIC %md
# MAGIC # Read in data

# COMMAND ----------

data_path = '/FileStore/tables/demand_forecasting_solution_accellerator/demand_df_delta/'
demand_df = spark \
  .read \
  .format("delta") \
  .load(data_path)

display(demand_df)

# COMMAND ----------

display(demand_df.select("Product").dropDuplicates())

# COMMAND ----------

display(demand_df.groupBy("Product").agg(F.countDistinct("SKU").alias("distinctSKUs")).dropDuplicates())

# COMMAND ----------

# MAGIC %md
# MAGIC The data consists of stacked time series. Each product has a number of SKUs and for each SKU there is demand for a given date. The demand time series for each product and SKU are consolidated to a one DataFrame. Each product group has a similar structure:
# MAGIC - There is a christmas effect with a significant drop in demand before christmas followed by an increase in demand after christmas
# MAGIC - At approximately the beginning of March in the year 2020, the demand drops to approximately 70%. 
# MAGIC - Before that, we observe a positive trend with decreasing increments. 
# MAGIC - After the Corona drop, there is a recovery phase with a clear positive trend. 
# MAGIC - The fluctuation of the time series, as well as their overall mean differ across products. 

# COMMAND ----------

for product_loop in demand_df.select("Product").distinct().collect():

  plot_pdf = demand_df.filter(col("Product") == product_loop[0]).select("Date", "Demand").toPandas()
  plt.subplots(figsize=(18,6))
  ax = seaborn.boxplot(x=plot_pdf["Date"], y=plot_pdf["Demand"])
  ax.xaxis.set_major_locator(md.WeekdayLocator(byweekday = 1))
  plt.setp(ax.xaxis.get_majorticklabels(), rotation = 90)
  ax.xaxis.set_minor_locator(md.DayLocator(interval = 1))
  plt.xlabel("Date")
  plt.ylabel("Demand")
  plt.title(product_loop[0])

# COMMAND ----------

# MAGIC %md
# MAGIC When looking at an arbitrary single time series we again see positive trend, followed by a drop in March 2020, followed by an increase. Next to random fluctuation we observe a christmas effect.

# COMMAND ----------

# Extract a signle time series and convert to pandas dataframe
pdf = demand_df.join(demand_df.sample(False, 1 / demand_df.count(), seed=0).limit(1).select("SKU"), on=["SKU"], how="inner").toPandas()
print(pdf)

# Create single series 
series_df = pd.Series(pdf['Demand'].values, index=pdf['Date'])
series_df = series_df.asfreq(freq='W-MON')
print(series_df)

# COMMAND ----------

display(pdf)

# COMMAND ----------

# MAGIC %md
# MAGIC # Data Preparation

# COMMAND ----------

# MAGIC %md
# MAGIC Let's assume that the last weeks of each time series is the furture that is going to be forecasted by the demand of former time series points.

# COMMAND ----------

is_history = [True] * (len(series_df) - forecast_horizon) + [False] * forecast_horizon

# COMMAND ----------

# MAGIC %md
# MAGIC A training and validation data set can be derived from this.

# COMMAND ----------

train = series_df.iloc[is_history]
score = series_df.iloc[~np.array(is_history)]

# COMMAND ----------

# MAGIC %md
# MAGIC Furthermore, we derive some exogenous dummy variables indicating before/after the pandemic started as well as the effect of Christmas.

# COMMAND ----------

exo_df = pdf.assign(Week = pd.DatetimeIndex(pdf["Date"]).isocalendar().week.tolist()) 

exo_df = exo_df \
  .assign(covid = np.where(pdf["Date"] >= np.datetime64(covid_breakpoint), 1, 0).tolist()) \
  .assign(christmas = np.where((exo_df["Week"] >= 51) & (exo_df["Week"] <= 52) , 1, 0).tolist()) \
  .assign(new_year = np.where((exo_df["Week"] >= 1) & (exo_df["Week"] <= 4)  , 1, 0).tolist()) \
  .set_index('Date')

exo_df = exo_df[["covid", "christmas", "new_year" ]]
exo_df = exo_df.asfreq(freq='W-MON')
print(exo_df)

# COMMAND ----------

# MAGIC %md
# MAGIC This dataset is then separated into a training and validation data set.

# COMMAND ----------

train_exo = exo_df.iloc[is_history]  
score_exo = exo_df.iloc[~np.array(is_history)]

# COMMAND ----------

# MAGIC %md
# MAGIC # Simple Exponential Smoothing

# COMMAND ----------

# MAGIC %md
# MAGIC Let's try a simple exponential smoothing model. We observe a good fit to adopt to the time series and its irregular components. In the forecasting horizon, we just see a constant trend. This model does not do a good job for forecasting the Christmas effect or the recovery period after the pandemic started.

# COMMAND ----------

model1 =  SimpleExpSmoothing(train, initialization_method="estimated")
fit1 = model1.fit(optimized=True)
fcast1 = fit1.forecast(forecast_horizon).rename("Simple Exponential Smoothing")

# COMMAND ----------

plt.figure(figsize=(18, 6))
(line0,) = plt.plot(series_df, marker="o", color="black")
plt.plot(fit1.fittedvalues, marker="o", color="blue")
(line1,) = plt.plot(fcast1, marker="o", color="blue")
plt.axvline(x = min(score.index.values), color = 'red', label = 'axvline - full height')
plt.legend([line0, line1], ["Actuals", fcast1.name])
plt.xlabel("Time")
plt.ylabel("Demand")
plt.title("Simple Exponential Smoothing")

# COMMAND ----------

# MAGIC %md
# MAGIC # Holt’s Method

# COMMAND ----------

# MAGIC %md
# MAGIC The Holt’s Method introduces different versions of a trend. We observe a linear trend that, despite optimal parameters, neither fits the true trend nor the Christmas effect.

# COMMAND ----------

fit1 = Holt(train, initialization_method="estimated").fit(optimized=True, method="ls")
fcast1 = fit1.forecast(forecast_horizon).rename("Holt's linear trend")

fit2 = Holt(train, exponential=True, initialization_method="estimated").fit(optimized=True, method="ls")
fcast2 = fit2.forecast(forecast_horizon).rename("Exponential trend")

fit3 = Holt(train, damped_trend=True, initialization_method="estimated").fit(optimized=True, method="ls")
fcast3 = fit3.forecast(forecast_horizon).rename("Additive damped trend")

plt.figure(figsize=(18, 6))
(line0,) = plt.plot(series_df, marker="o", color="black")
plt.plot(fit1.fittedvalues, color="blue")
(line1,) = plt.plot(fcast1, marker="o", color="blue")
plt.plot(fit2.fittedvalues, color="red")
(line2,) = plt.plot(fcast2, marker="o", color="red")
plt.plot(fit3.fittedvalues, color="green")
(line3,) = plt.plot(fcast3, marker="o", color="green")
plt.axvline(x = min(score.index.values), color = 'red', label = 'axvline - full height')
plt.legend([line0, line1, line2, line3], ["Actuals", fcast1.name, fcast2.name, fcast3.name])
plt.xlabel("Time")
plt.ylabel("Demand")
plt.title("Holt's Method")

# COMMAND ----------

# MAGIC %md
# MAGIC # Holt’s Winters Seasonal

# COMMAND ----------

# MAGIC %md
# MAGIC The Holt's Winters Seasonal method additionally adds a seasonal component. Note that the actuals have two seasonal components. The ongoing seasonal component can be fit in the forecasting horizon. However, this class of models is not able to incorporate the Christmas effect. 

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
plt.plot(series_df, marker="o", color="black")
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
# MAGIC # SARIMAX

# COMMAND ----------

# MAGIC %md
# MAGIC A SARIMAX model allows to incorporate explanatory variables. From a business point of view, this helps to incorporate business knowledge about demand driving events. This could not only be a christmas effect, but also promotion actions. We observe that the model does a poor job when not taking advantage of the business knowledge. However, if incorporating exogenous variables, the Christmas effect and the after-pandemic trend can fit well in the forecasting horizon.

# COMMAND ----------

fit1 = SARIMAX(train, exog=train_exo, order=(2, 3, 3), seasonal_order=(0, 0, 0, 0), initialization_method="estimated").fit(warn_convergence = False)
fcast1 = fit1.predict(start = min(train.index), end = max(score_exo.index), exog = score_exo).rename("With exogenous variables")

fit2 = SARIMAX(train, order=(2, 3, 3), seasonal_order=(0, 0, 0, 0), initialization_method="estimated").fit(warn_convergence = False)
fcast2 = fit2.predict(start = min(train.index), end = max(score_exo.index)).rename("Without exogenous variables")

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
# MAGIC # Taking advantage of MLFlow and Hyperopt to find optimal parameters in the SARIMAX model

# COMMAND ----------

# MAGIC %md
# MAGIC For the model above, we applied a manual trial-and-error method to find good parameters. MLFlow and Hyperopt can be leveraged to find optimal parameters automatically. 

# COMMAND ----------

# MAGIC %md
# MAGIC First, we define an evaluation function. It trains a SARIMAX model with given parameters and evaluates it by calculating the mean squared error.

# COMMAND ----------

def evaluate_model(hyperopt_params):
  
  # Configure model parameters
  params = hyperopt_params
  
  assert "p" in params and "d" in params and "q" in params, "Please provide p, d, and q"
  
  if 'p' in params: params['p']=int(params['p']) # hyperopt supplies values as float but model requires int
  if 'd' in params: params['d']=int(params['d']) # hyperopt supplies values as float but model requires int
  if 'q' in params: params['q']=int(params['q']) # hyperopt supplies values as float but model requires int
    
  order_parameters = (params['p'],params['d'],params['q'])

  # For simplicity in this example, assume no seasonality
  model1 = SARIMAX(train, exog=train_exo, order=order_parameters, seasonal_order=(0, 0, 0, 0))
  fit1 = model1.fit(disp=False)
  fcast1 = fit1.predict(start = min(score_exo.index), end = max(score_exo.index), exog = score_exo )

  return {'status': hyperopt.STATUS_OK, 'loss': np.power(score.to_numpy() - fcast1.to_numpy(), 2).mean()}

# COMMAND ----------

# MAGIC %md
# MAGIC Second, we define a search space of parameters for which the model will be evaluated.

# COMMAND ----------

space = {
  'p': scope.int(hyperopt.hp.quniform('p', 0, 4, 1)),
  'd': scope.int(hyperopt.hp.quniform('d', 0, 4, 1)),
  'q': scope.int(hyperopt.hp.quniform('q', 0, 4, 1)) 
}

# COMMAND ----------

# MAGIC %md
# MAGIC We now take advantage of the search space and the evaluation function to automatically find optimal parameters. Note that these models get automatically tracked in an experiment.

# COMMAND ----------

with mlflow.start_run(run_name='mkh_test_sa'):
  argmin = fmin(
    fn=evaluate_model,
    space=space,
    algo=tpe.suggest,  # algorithm controlling how hyperopt navigates the search space
    max_evals=30,
    trials=SparkTrials(parallelism=1),
    verbose=True
    )

# COMMAND ----------

# MAGIC %md
# MAGIC By this means the optimal set of parameters can be found.

# COMMAND ----------

argmin
