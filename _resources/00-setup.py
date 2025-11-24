# Databricks notebook source
#dbutils.widgets.dropdown('reset_all_data', 'false', ['true', 'false'], 'Reset all data')
#dbutils.widgets.text('catalogName',  'maxkoehler_demos' , 'Catalog Name')
#dbutils.widgets.text('dbName',  'demand_db' , 'Database Name')

# COMMAND ----------

import os
import re 

# COMMAND ----------

catalogName = dbutils.widgets.get('catalogName')
dbName_prefix = dbutils.widgets.get('dbName')
reset_all_data = dbutils.widgets.get('reset_all_data') == 'true'

# COMMAND ----------

# Append user to dbName
from pyspark.sql.functions import regexp_extract
df = spark.sql("SELECT session_user() as user_name")
df = df.withColumn("name_before_at", regexp_extract("user_name", r"([^@]+)", 1))
user_name = df.select("name_before_at").first()["name_before_at"]
user_name = user_name.replace('.', '')
dbName = dbName_prefix + "_" + user_name
print(dbName)

# COMMAND ----------

print("Starting ./_resources/00-setup")

# COMMAND ----------

spark.sql(f"""USE CATALOG {catalogName}""")

# COMMAND ----------

if reset_all_data:
  spark.sql(f"DROP SCHEMA IF EXISTS {dbName} CASCADE")

# COMMAND ----------



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

def generate_data():
  dbutils.notebook.run(generate_data_notebook_path, 3000, {"reset_all_data": reset_all_data, "catalogName": catalogName,   "dbName": dbName})

# COMMAND ----------

if reset_all_data:
  print("Generating data...")
  generate_data()
  print("Data generated.")

# COMMAND ----------

#current_user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
#if current_user.rfind('@') > 0:
#  current_user_no_at = current_user[:current_user.rfind('@')]
#else:
#  current_user_no_at = current_user
#current_user_no_at = re.sub(r'\W+', '_', current_user_no_at)
#username_friendly = current_user_no_at
#display(current_user_no_at)

# COMMAND ----------

#mlflow.set_experiment('/Users/{}/supply_chain_optimization'.format(current_user))

# COMMAND ----------

#%run ../_resources_outside/00-global-setup $reset_all_data=$reset_all_data $db_prefix=demand_level_forecasting

# COMMAND ----------

print("Write config file..")

# COMMAND ----------

if os.path.exists("config.py"):
    os.remove("config.py")

# COMMAND ----------

config = f"""# Databricks notebook source
# MAGIC %md
# MAGIC ### Set-up notebook config
# COMMAND ----------
%sql
USE CATALOG {catalogName};
USE SCHEMA {dbName};
"""

with open("config.py", "w") as f:
    f.write(config)

# COMMAND ----------

print("Config file written.")

# COMMAND ----------

print("Ending ./_resources/00-setup")
