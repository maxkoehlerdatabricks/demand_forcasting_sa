# Databricks notebook source
dbutils.widgets.text('dbName',  'demand_db' , 'Database Name')
dbutils.widgets.text('cloud_storage_path',  '/FileStore/tables/demand_forecasting_solution_accelerator/', 'Storage Path')

# COMMAND ----------

dbName = dbutils.widgets.get('dbName')
cloud_storage_path = dbutils.widgets.get('cloud_storage_path')

# COMMAND ----------

#notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()

# COMMAND ----------

#notebook_path

# COMMAND ----------

#dbutils.notebook.run(generate_data_notebook_path, 600, {"reset_all_data": reset_all, "dbName": dbName, "cloud_storage_path": cloud_storage_path})

# COMMAND ----------

#dbutils.widgets.removeAll()

# COMMAND ----------

#if os.path.basename(os.path.dirname(notebook_path)) == 'Helper':
#  %run ../_resources_outside/00-global-setup $reset_all_data=false $db_prefix=demand_level_forecasting
#else:
#  %run ./_resources_outside/00-global-setup $reset_all_data=false $db_prefix=demand_level_forecasting 

# COMMAND ----------

import os
import random
import pyspark.sql.functions as f
from pyspark.sql.types import FloatType

# COMMAND ----------

demand_raw_df = spark.read.table(f"{dbName}.forecast_raw")
all_skus = demand_raw_df.select('SKU').distinct().rdd.flatMap(lambda x: x).collect()
material_shortages_sku = random.sample(set(all_skus), 2)
all_raw =  demand_raw_df.filter(f.col("SKU").isin(material_shortages_sku)).select('RAW').distinct().rdd.flatMap(lambda x: x).collect()
material_shortages_raw = random.sample(set(all_raw), 3)
maximum_date =  max(demand_raw_df.select('Date').distinct().rdd.flatMap(lambda x: x).collect())

# COMMAND ----------

def random_fraction(z):
  return(random.uniform(0.5, 0.9))

random_fractionUDF = udf(lambda z: random_fraction(z),FloatType())

# COMMAND ----------

material_shortage_df = (demand_raw_df.
                          filter((f.col("SKU").isin(material_shortages_sku)) & (f.col("Date") == maximum_date)  & (f.col("RAW").isin(material_shortages_raw))).
                          withColumn("fraction", random_fractionUDF(  f.col("Demand_Raw")  )).
                          withColumn("available_demand", f.floor(f.col("fraction") * f.col("Demand_Raw"))).
                          select("RAW", "Date", "available_demand") 
 )
#display(material_shortage_df)

# COMMAND ----------

material_shortage_df_delta_path = os.path.join(cloud_storage_path, 'material_shortage')

# COMMAND ----------

# Write the data 
material_shortage_df.write \
.mode("overwrite") \
.format("delta") \
.save(material_shortage_df_delta_path)

# COMMAND ----------

spark.sql(f"DROP TABLE IF EXISTS {dbName}.material_shortage")
spark.sql(f"CREATE TABLE {dbName}.material_shortage USING DELTA LOCATION '{material_shortage_df_delta_path}'")
