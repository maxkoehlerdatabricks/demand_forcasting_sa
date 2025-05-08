# Databricks notebook source
# MAGIC %md
# MAGIC # Does not work properly -> Only linear forecast ⚠️ 

# COMMAND ----------

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
# MAGIC - Hyperopt can also perform hyperparameter tuning from within a Pandas UDF  

# COMMAND ----------

#If True, all output files are in user specific databases, If False, a global database for the report is used
#user_based_data = True

# COMMAND ----------

# %run ./_resources_outside/00-global-setup $reset_all_data=false $db_prefix=demand_level_forecasting

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

# MAGIC %md
# MAGIC
# MAGIC ## Using Databricks SQL AI Forecast Function
# MAGIC ai_forecast() is a table-valued function designed to extrapolate time series data into the future
# MAGIC
# MAGIC [AI_Forecast Reference](http://docs.databricks.com/aws/en/sql/language-manual/functions/ai_forecast)
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ####Requirement
# MAGIC - Pro or Serverless SQL warehouse
# MAGIC - In Databricks Runtime 15.1 and above, this function is supported in Databricks notebooks, including notebooks that are run as a task in a Databricks workflow.
# MAGIC - For batch inference workloads, Databricks recommends Databricks Runtime 15.4 ML LTS for improved performance.

# COMMAND ----------

# MAGIC %md
# MAGIC Comment Patrick: Only worked for me with using SQL Serverless Warehouse. With normal Serverless failed with FEATURE NOT ENABLED error

# COMMAND ----------

# MAGIC %md
# MAGIC ###Check AI_Forecast() Function Capabilities
# MAGIC Using `AI_Forecast()` Function to forecast historical data for one SKU to show forecast quality 

# COMMAND ----------

# MAGIC %sql
# MAGIC WITH input_data AS (
# MAGIC   SELECT Date, Demand, SKU FROM ${catalogName}.${dbName}.part_level_demand
# MAGIC   WHERE SKU = (SELECT SKU FROM ${catalogName}.${dbName}.part_level_demand LIMIT 1) 
# MAGIC   AND Date <= (SELECT DATE_ADD(MIN(Date), 365*2) FROM ${catalogName}.${dbName}.part_level_demand)
# MAGIC )
# MAGIC SELECT forecast.Date, forecast.SKU, Demand, Demand_Forecast FROM AI_FORECAST(
# MAGIC     TABLE(input_data),
# MAGIC     horizon => DATE_ADD(DAY, 40, (SELECT MAX(Date) FROM ${catalogName}.${dbName}.part_level_demand)) ,
# MAGIC     time_col => 'Date',
# MAGIC     value_col => 'Demand',
# MAGIC     group_col => 'SKU',
# MAGIC     frequency => 'week'
# MAGIC ) AS forecast
# MAGIC LEFT JOIN ${catalogName}.${dbName}.part_level_demand AS demand
# MAGIC ON forecast.Date = Demand.Date AND forecast.SKU = demand.SKU 
# MAGIC ORDER BY Date ASC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Make demand forecasts using AI_Forecast() function
# MAGIC
# MAGIC Apply the `AI_Forecast()` function to our whole dataset to generate demand forecasts for the next 40 days and store the results into a delta table.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE ${catalogName}.${dbName}.part_level_demand_forecasts_only AS
# MAGIC SELECT * FROM AI_FORECAST(
# MAGIC     TABLE(SELECT Date, Demand, SKU FROM ${catalogName}.${dbName}.part_level_demand),
# MAGIC     horizon => DATE_ADD(DAY, 40, (SELECT MAX(Date) FROM ${catalogName}.${dbName}.part_level_demand)) ,
# MAGIC     time_col => 'Date',
# MAGIC     value_col => 'Demand',
# MAGIC     group_col => 'SKU',
# MAGIC     frequency => 'week'
# MAGIC )
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM ${catalogName}.${dbName}.part_level_demand_forecasts_only
