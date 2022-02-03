# Databricks notebook source
# MAGIC %md
# MAGIC # Packages and Parameters

# COMMAND ----------

import os
import datetime
from random import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from statsmodels.tsa.statespace.sarimax import SARIMAX

import mlflow
import hyperopt
from hyperopt import hp, fmin, tpe, SparkTrials, STATUS_OK, space_eval
from hyperopt.pyll.base import scope
mlflow.autolog(disable=True)


%matplotlib inline

forecast_horizon = 40

# COMMAND ----------

# MAGIC %md
# MAGIC for the models see e.g.
# MAGIC https://www.statsmodels.org/stable/examples/notebooks/generated/exponential_smoothing.html

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

# Extract a signle time series and convert to pandas dataframe
pdf = demand_df.join(demand_df.sample(False, 1 / demand_df.count(), seed=0).limit(1).select("SKU"), on=["SKU"], how="inner").toPandas()
print(pdf)

# Create single series 
series_df= pd.Series(pdf['Demand'].values, index=pdf['Date'])
series_df = series_df.asfreq(freq='W-MON')
print(series_df)

# COMMAND ----------

corona_breakpoint = datetime.date(year=2020,month=3,day=1)

exo_df = pdf.assign(Week = pd.DatetimeIndex(pdf["Date"]).isocalendar().week.tolist()) 

exo_df = exo_df \
  .assign(AfterCorona = np.where(pdf["Date"] >= np.datetime64(corona_breakpoint), 1, 0).tolist()  ) \
  .assign(BeforeXMas = np.where( (exo_df["Week"] >= 51) & (exo_df["Week"] <= 52) , 1, 0).tolist()) \
  .assign(AfterXMas = np.where( (exo_df["Week"] >= 1) & (exo_df["Week"] <= 4)  , 1, 0).tolist()) \
  .set_index('Date')

exo_df = exo_df[["AfterCorona", "BeforeXMas", "AfterXMas" ]]
exo_df = exo_df.asfreq(freq='W-MON')
print(exo_df)

# COMMAND ----------

is_hsitory = [True] * (len(series_df) - forecast_horizon) + [False] * forecast_horizon
print(is_hsitory)

# COMMAND ----------

train = series_df.iloc[is_hsitory]
score = series_df.iloc[~np.array(is_hsitory)]
train_exo = exo_df.iloc[is_hsitory]  
score_exo = exo_df.iloc[~np.array(is_hsitory)]

# COMMAND ----------

# MAGIC %md
# MAGIC # Simple Exponential Smoothing 
# MAGIC SES is able to fit the curve well, howver: only flat forecasts

# COMMAND ----------

model1 =  SimpleExpSmoothing(train, initialization_method="estimated")
fit1 = model1.fit(optimized=True)
fcast1 = fit1.forecast(forecast_horizon).rename("Simple Exponential Smoothing")
print(fcast1)

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
# MAGIC # Holts method
# MAGIC Holts method is able to deal with a trend. A damped trend often better. 

# COMMAND ----------

fit1 = Holt(train, initialization_method="estimated").fit(optimized=True)
fcast1 = fit1.forecast(forecast_horizon).rename("Holt's linear trend")

fit2 = Holt(train, exponential=True, initialization_method="estimated").fit(optimized=True)
fcast2 = fit2.forecast(forecast_horizon).rename("Exponential trend")

fit3 = Holt(train, damped_trend=True, initialization_method="estimated").fit(optimized=True)
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
# MAGIC # Holts Winters Seasonal method

# COMMAND ----------

fit1 = ExponentialSmoothing(
    train,
    seasonal_periods=3,
    trend="add",
    seasonal="add",
    use_boxcox=True,
    initialization_method="estimated",
).fit()

fcast1 = fit1.forecast(forecast_horizon).rename("Additive trend and additive seasonal")

fit2 = ExponentialSmoothing(
    train,
    seasonal_periods=4,
    trend="add",
    seasonal="mul",
    use_boxcox=True,
    initialization_method="estimated",
).fit()

fcast2 = fit2.forecast(forecast_horizon).rename("Additive trend and multiplicative seasonal")

fit3 = ExponentialSmoothing(
    train,
    seasonal_periods=4,
    trend="add",
    seasonal="add",
    damped_trend=True,
    use_boxcox=True,
    initialization_method="estimated",
).fit()

fcast3 = fit3.forecast(forecast_horizon).rename("Additive damped trend and additive seasonal")

fit4 = ExponentialSmoothing(
    train,
    seasonal_periods=4,
    trend="add",
    seasonal="mul",
    damped_trend=True,
    use_boxcox=True,
    initialization_method="estimated",
).fit()


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
plt.title("SHolts Winters Seasonal Method")
                                                

# COMMAND ----------

# MAGIC %md
# MAGIC # Try with SARIMAX

# COMMAND ----------

fit1 = SARIMAX(train, exog=train_exo, order=(2, 3, 2), seasonal_order=(0, 0, 0, 0), initialization_method="estimated").fit()
fcast1 = fit1.predict(start = min(train.index), end = max(score_exo.index), exog = score_exo )

fit2 = SARIMAX(train, order=(2, 3, 2), seasonal_order=(0, 0, 0, 0), initialization_method="estimated").fit()
fcast2 = fit2.predict(start = min(train.index), end = max(score_exo.index), exog = score_exo )

# COMMAND ----------

plt.figure(figsize=(18, 6))
plt.plot(series_df, marker="o", color="black")
plt.plot(fcast1[10:], color="blue")
(line1,) = plt.plot(fcast1[10:], marker="o", color="blue")
plt.plot(fcast2[10:], color="green")
(line1,) = plt.plot(fcast2[10:], marker="o", color="green")

plt.axvline(x = min(score.index.values), color = 'red', label = 'axvline - full height')

# COMMAND ----------

# MAGIC %md
# MAGIC # Implememnt a grid search in mlflow with Sarimax

# COMMAND ----------

def evaluate_model(hyperopt_params):
  
  # configure model parameters
  params = hyperopt_params
  
  assert "p" in params and "d" in params and "q" in params, "Please prvide p,d and q"
  
  if 'p' in params: params['p']=int(params['p'])   # hyperopt supplies values as float but must be int
  if 'd' in params: params['d']=int(params['d']) # hyperopt supplies values as float but must be int
  if 'q' in params: params['q']=int(params['q']) # hyperopt supplies values as float but must be int
    
  order_parameters = (params['p'],params['d'],params['q'])

  model1 = SARIMAX(train, exog= train_exo, order=order_parameters, seasonal_order=(0, 0, 0, 0))
  fit1 = model1.fit(disp=False)
  fcast1 = fit1.predict(start = min(score_exo.index), end = max(score_exo.index), exog = score_exo )

  return {'status': hyperopt.STATUS_OK, 'loss': np.power(score.to_numpy() - fcast1.to_numpy(), 2).mean()}
  

# COMMAND ----------

space = {
  'p': scope.int(hyperopt.hp.quniform('p', 0, 4, 1)),
  'd': scope.int(hyperopt.hp.quniform('d', 0, 4, 1)),
  'q': scope.int(hyperopt.hp.quniform('q', 0, 4, 1))
    
}

# COMMAND ----------

with mlflow.start_run(run_name='mkh_test_sa'):
  argmin = fmin(
    fn=evaluate_model,
    space=space,
    algo=tpe.suggest,  # algorithm controlling how hyperopt navigates the search space
    max_evals=3,
    trials=SparkTrials(parallelism=1),
    verbose=True
    )

# COMMAND ----------

#Choose best model
model1 = SARIMAX(train, exog= train_exo, order=(argmin.get('d'), argmin.get('p'), argmin.get('q')), seasonal_order=(0, 0, 0, 0))
fit1 = model1.fit(disp=False)
fcast1 = fit1.predict(start = min(score_exo.index), end = max(score_exo.index), exog = score_exo )

#And score
fcst_df = pdf.iloc[~np.array(is_hsitory)].assign(Demand = fcast1.tolist())
history_df =  pdf.iloc[is_hsitory]
pdf_fcst =  history_df.append(fcst_df)
pdf_fcst

# COMMAND ----------

# MAGIC %md
# MAGIC # Implement a grid search in mlflow with Sarimax and 

# COMMAND ----------

#Define objective function

def evaluate_model(hyperopt_params):
  
  # configure model parameters
  params = hyperopt_params
  
  assert "type" in params
  assert params["type"] in [ "ses", "holt_linear_trend", "holt_exponential_trend",  "holt_additive_dumped_trend",  "sarimax"]
  
  model_type = params["type"] 
  
  
  train = series_df.iloc[is_hsitory]
  score = series_df.iloc[~np.array(is_hsitory)]
  train_exo = exo_df.iloc[is_hsitory]  
  score_exo = exo_df.iloc[~np.array(is_hsitory)]
  
  
  if model_type == "ses":
    model_ses =  SimpleExpSmoothing(train, initialization_method="estimated")
    fit_ses = model_ses.fit(optimized=True)
    forecast = fit_ses.forecast(forecast_horizon)
  elif model_type == "holt_linear_trend":
    fit_holt_linear_trend = Holt(series_df, initialization_method="estimated").fit(optimized=True, method='ls')
    forecast = fit_holt_linear_trend.forecast(forecast_horizon)
  elif model_type == "holt_exponential_trend":
    fit_exponential_trend = Holt(series_df, exponential=True, initialization_method="estimated").fit(optimized=True, method='ls')
    forecast = fit_exponential_trend.forecast(forecast_horizon)
  elif model_type == "holt_additive_dumped_trend":
    fit_additive_dumped_trend = Holt(series_df, damped_trend=True, initialization_method="estimated").fit(optimized=True, method='ls')
    forecast = fit_additive_dumped_trend.forecast(forecast_horizon)
  else:
    assert "p" in params and "d" in params and "q"
    order_parameters = (params['p'],params['d'],params['q'])
    fit_model_sarimax = SARIMAX(train, exog= train_exo, order=order_parameters, seasonal_order=(0, 0, 0, 0)).fit(disp=False)
    forecast = fit_model_sarimax.predict(start = min(score_exo.index), end = max(score_exo.index), exog = score_exo )
    
  loss = np.power(score.to_numpy() - forecast.to_numpy(), 2).mean()
   
  return {'status': hyperopt.STATUS_OK, 'loss': loss}

# COMMAND ----------

space = hp.choice("type", [
    {
        "type": "ses"
    },
    {
        "type": "holt_linear_trend"
    },
    {
        "type": "holt_exponential_trend"
    },
    {
        "type": "holt_additive_dumped_trend"
    },
    {
        "type": "sarimax",
        "p": scope.int(hyperopt.hp.quniform("p", 0, 4, 1)),
        "d": scope.int(hyperopt.hp.quniform("d", 0, 4, 1)),
        "q": scope.int(hyperopt.hp.quniform("q", 0, 4, 1))
    }
])

# COMMAND ----------

#I did not succeed in not tracking experiments, mlflow.autolog(disable=True) didi not work, neither did with commenting mlflow.start_run(run_name='mkh_test_sa_2'):

with mlflow.start_run(run_name='mkh_test_sa_2'):
  
  argmin = fmin(
    fn=evaluate_model,
    space=space,
    algo=tpe.suggest,  # algorithm controlling how hyperopt navigates the search space
    max_evals=3,
    trials=SparkTrials(parallelism=1), # THIS VALUE IS ALIGNED WITH THE NUMBER OF WORKERS IN MY GPU-ENABLED CLUSTER (guidance differs for CPU-based clusters)
    verbose=True
  )
    
mlflow.end_run()


# COMMAND ----------

print(hyperopt.space_eval(space, argmin))

# COMMAND ----------

# retrain and log model separately

# COMMAND ----------

argmin

# COMMAND ----------


