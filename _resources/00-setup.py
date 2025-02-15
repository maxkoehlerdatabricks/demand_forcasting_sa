# Databricks notebook source
# MAGIC %pip install databricks-sdk --upgrade
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.dropdown('reset_all_data', 'false', ['true', 'false'], 'Reset all data')
dbutils.widgets.text('catalogName',  'maxkoehler_demos' , 'Catalog Name')
dbutils.widgets.text('dbName',  'demand_db' , 'Database Name')

# COMMAND ----------

catalogName = dbutils.widgets.get('catalogName')
dbName = dbutils.widgets.get('dbName')
reset_all_data = dbutils.widgets.get('reset_all_data') == 'true'

# COMMAND ----------

import os
import re 
import mlflow

# COMMAND ----------

print("Starting ./_resources/00-setup")

# COMMAND ----------

if reset_all_data:
  spark.sql(f"DROP CATALOG IF EXISTS {catalogName} CASCADE")

spark.sql(f"""create catalog if not exists {catalogName}""")
spark.sql(f"""USE CATALOG {catalogName}""")
spark.sql(f"""create database if not exists {dbName}""")
spark.sql(f"""USE {dbName}""")

# COMMAND ----------

print(f"The catalog {catalogName} will be used")
print(f"The database {dbName} will be used")

# COMMAND ----------

dirname = os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
filename = "01-data-generator"
if (os.path.basename(dirname) != '_resources'):
  dirname = os.path.join(dirname,'_resources')

generate_data_notebook_path = os.path.join(dirname,filename)

# print(generate_data_notebook_path)

def generate_data():
  dbutils.notebook.run(generate_data_notebook_path, 3000, {"reset_all_data": reset_all_data, "catalogName": catalogName,   "dbName": dbName})

# COMMAND ----------

if reset_all_data:
  generate_data()

# COMMAND ----------

current_user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
if current_user.rfind('@') > 0:
  current_user_no_at = current_user[:current_user.rfind('@')]
else:
  current_user_no_at = current_user
current_user_no_at = re.sub(r'\W+', '_', current_user_no_at)
username_friendly = current_user_no_at
display(current_user_no_at)

# COMMAND ----------

mlflow.set_experiment('/Users/{}/supply_chain_optimization'.format(current_user))

# COMMAND ----------

#%run ../_resources_outside/00-global-setup $reset_all_data=$reset_all_data $db_prefix=demand_level_forecasting

# COMMAND ----------

print("Ending ./_resources/00-setup")
