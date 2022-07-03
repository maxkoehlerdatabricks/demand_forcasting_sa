# Databricks notebook source
dbutils.widgets.dropdown("reset_all_data", "false", ["true", "false"], "Reset all data")

# COMMAND ----------

#If True, all output files are in user specific databases, If False, a global database for the report is used
user_based_data = True

# COMMAND ----------

# MAGIC %run ../_resources_outside/00-global-setup $reset_all_data=$reset_all_data $db_prefix=demand_level_forecasting

# COMMAND ----------

if (not user_based_data):
  cloud_storage_path = '/FileStore/tables/demand_forecasting_solution_accelerator/'
  dbName = 'demand_db' 

# COMMAND ----------

 #creates hive_metastore.demand_level_forecasting_max_kohler
dbName

# COMMAND ----------

cloud_storage_path
