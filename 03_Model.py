# Databricks notebook source
# MAGIC %md
# MAGIC References:
# MAGIC * https://databricks.com/notebooks/simple-aws/petastorm-spark-converter-pytorch.html
# MAGIC * https://databricks.com/notebooks/simple-aws/petastorm-spark-converter-tensorflow.html
# MAGIC * https://docs.microsoft.com/en-us/learn/modules/deep-learning-with-horovod-distributed-training/

# COMMAND ----------

import pandas as pd
import numpy as np
import datetime

from statsmodels.tsa.statespace.sarimax import SARIMAX

forecast_horizon = 40
corona_breakpoint = datetime.date(year=2020,month=3,day=1)

# COMMAND ----------

# Read in data
data_path = '/FileStore/tables/demand_forecasting_solution_accellerator/demand_df_delta/'
demand_df = spark \
  .read \
  .format("delta") \
  .load(data_path)

display(demand_df)

# COMMAND ----------

# Extract a single time series and convert to pandas dataframe
pdf = demand_df.join(demand_df.sample(False, 1 / demand_df.count(), seed=0).limit(1).select("SKU"), on=["SKU"], how="inner").toPandas()
print(pdf)

# Create single series 
series_df= pd.Series(pdf['Demand'].values, index=pdf['Date'])
series_df = series_df.asfreq(freq='W-MON')
print(series_df)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC # Train all

# COMMAND ----------

def plot_forecast(series_df, fcast1, fcast2):
  plt.figure(figsize=(18, 6))
  plt.plot(series_df, marker="o", color="black")
  plt.plot(fcast1[10:], color="blue")
  (line1,) = plt.plot(fcast1[10:], marker="o", color="blue")
  plt.plot(fcast2[10:], color="green")
  (line1,) = plt.plot(fcast2[10:], marker="o", color="green")

  plt.axvline(x = min(score.index.values), color = 'red', label = 'axvline - full height')
  plt.legend([line0, line1, line2], ["Actuals", fcast1.name, fcast2.name])
  plt.xlabel("Time")
  plt.ylabel("Demand")
  plt.title("SARIMAX")
  

def exogenous_var(pdf):
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

# Get unique SKUs
unique_SKUs = demand_df.select('SKU').distinct().collect()

# Loop through unique SKUs
for row in unique_SKUs:
  SKU = row['SKU']
    
  pandas_df = demand_df.filter(demand_df.SKU == SKU).toPandas()
  print(pandas_df)

  # Create single series 
  series_df= pd.Series(pandas_df['Demand'].values, index=pandas_df['Date'])
  series_df = series_df.asfreq(freq='W-MON')
  #print(series_df)

  is_hsitory = [True] * (len(series_df) - forecast_horizon) + [False] * forecast_horizon

  train = series_df.iloc[is_hsitory]
  score = series_df.iloc[~np.array(is_hsitory)]
  print('Training data set size : {}'.format(train.size))
  print('Scoring data set size  : {}'.format(score.size))

  exo_df = exogenous_var(pandas_df)
  train_exo = exo_df.iloc[is_hsitory]  
  score_exo = exo_df.iloc[~np.array(is_hsitory)]

  fit1 = SARIMAX(train, exog=train_exo, order=(2, 3, 2), seasonal_order=(0, 0, 0, 0), initialization_method="estimated").fit(warn_convergence = False)
  fcast1 = fit1.predict(start = min(train.index), end = max(score_exo.index), exog = score_exo ).rename("With exogenous variables")

  fit2 = SARIMAX(train, order=(2, 3, 2), seasonal_order=(0, 0, 0, 0), initialization_method="estimated").fit(warn_convergence = False)
  fcast2 = fit2.predict(start = min(train.index), end = max(score_exo.index), exog = score_exo ).rename("Without exogenous variables")


  break #REMOVE

# COMMAND ----------

#pandas_df = demand_df.select('SKU', 'LRR_UOOZDJ').toPandas()
score.size

# COMMAND ----------


