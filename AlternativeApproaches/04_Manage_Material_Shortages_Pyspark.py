# Databricks notebook source
# MAGIC %md
# MAGIC # Manage a material shortage
# MAGIC After checking with the supplier how much raw material can actually be delivered we can now traverse the manufacturing value chain forwards to find out how many SKU's can actually be shipped to the customer. If one raw material is the bottleneck for producing a specific SKU, orders of the other raw materials for that SKU can be adjusted accordingly to save storage costs.

# COMMAND ----------

# MAGIC %md
# MAGIC *Prerequisite: Make sure to run 01_Introduction_And_Setup, 02_Fine_Grained_Demand_Forecasting and 03_Derive_Raw_Material_Demand before running this notebook.*
# MAGIC
# MAGIC While the previous notebook *(03_Derive_Raw_Material_Demand)* demonstrated Databricks' graph functionality to traverse the manufacturing value chain backwards to find out how much raw material is needed for production, this notebook:
# MAGIC - Checks the availability of each raw material
# MAGIC - Traverses the manufacturing value chain forwards to check the quantity of SKU's that can actually be delivered
# MAGIC - Adjusts orders for raw materials accordingly
# MAGIC
# MAGIC Key highlights for this notebook:
# MAGIC - Use Delta and the previous notebook's results to traverse the manufacturing value chain forwards

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://raw.githubusercontent.com/maxkoehlerdatabricks/demand_forcasting_sa/max/Pictures/operations_process_backwards.png" width="1500"/>

# COMMAND ----------

#If True, all output files are in user specific databases, If False, a global database for the report is used
#user_based_data = True

# COMMAND ----------

#%run ./_resources_outside/00-global-setup $reset_all_data=false $db_prefix=demand_level_forecasting

# COMMAND ----------

#if (not user_based_data):
#  cloud_storage_path = '/FileStore/tables/demand_forecasting_solution_accelerator/'
#  dbName = 'demand_db' 
  
#print(cloud_storage_path)
#print(dbName)

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

import os
import pyspark.sql.functions as f

# COMMAND ----------

# MAGIC %md
# MAGIC ## Get all reported raw material shortages

# COMMAND ----------

notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
notebook_path = os.path.join(os.path.dirname(notebook_path),"Helper/Simulate_Material_Shortages")
notebook_path

# COMMAND ----------

dbutils.notebook.run(notebook_path, 600, {"catalogName" : catalogName,"dbName": dbName})

# COMMAND ----------

display(spark.read.table(f"{catalogName}.{dbName}.material_shortage"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Get the affected SKU's
# MAGIC Get the affected SKU's and derive the quantity that can actually be shipped to the customer

# COMMAND ----------

demand_raw_df = spark.read.table(f"{catalogName}.{dbName}.forecast_raw")
material_shortage_df = spark.read.table(f"{catalogName}.{dbName}.material_shortage")

# COMMAND ----------

affected_skus_df = (material_shortage_df.
                    join(demand_raw_df, ["RAW","Date"], how="inner").
                    withColumn("fraction_raw", f.col("available_demand") / f.col("Demand_Raw"))
                   )
                    
min_fraction =  (affected_skus_df.
                    groupBy("SKU").
                    agg(f.min(f.col("fraction_raw")).alias("min_fraction_raw"))
                   )

affected_skus_df = (
                  affected_skus_df.
                  join(min_fraction, ["SKU"]).
                  withColumnRenamed("available_demand", "Adjusted_Demand_Raw").
                  withColumn("Adjusted_Demand_SKU", f.floor(f.col("Demand_SKU") * f.col("min_fraction_raw")) ).
                  select(f.col("RAW"), f.col("Date"), f.col("SKU").alias("Affetced_SKU"), f.col("Product").alias("Affected_Product"), f.col("Demand_RAW"), f.col("Adjusted_Demand_Raw"), f.col("Demand_SKU"), f.col("Adjusted_Demand_SKU"), f.col("min_fraction_raw").alias("Available_Fraction_For_SKU"))
)

display(affected_skus_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Get the amount of overplanning other raw materials
# MAGIC If one raw material has a shortage, other raw materials in the specific SKU are overplanned and orders can be adjusted to reduce storage costs

# COMMAND ----------

raw_overplanning_df = (affected_skus_df.
                        select(f.col("Affetced_SKU").alias("SKU"), f.col("Date"), f.col("Available_Fraction_For_SKU")).
                        join(demand_raw_df, ["SKU", "Date"], how="inner").
                        withColumn("Demand_Raw_Adjusted", f.floor(f.col("Demand_RAW") * f.col("Available_Fraction_For_SKU"))).
                        select(f.col("RAW"), f.col("Date"), f.col("Demand_Raw"), f.col("Demand_Raw_Adjusted"))
                      )


display(raw_overplanning_df)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Save to Delta

# COMMAND ----------

material_shortage_data_path = os.path.join(cloud_storage_path, "material_shortage_sku")

# COMMAND ----------

# Write the data 
affected_skus_df.write \
.mode("overwrite") \
.format("delta") \
.save(material_shortage_data_path)

# COMMAND ----------

spark.sql(f"DROP TABLE IF EXISTS {dbName}.material_shortage_sku")
spark.sql(f"CREATE TABLE {dbName}.material_shortage_sku USING DELTA LOCATION '{material_shortage_data_path}'")

# COMMAND ----------

material_shortage_raw_data_path = os.path.join(cloud_storage_path, "material_shortage_raw")

# COMMAND ----------

# Write the data 
raw_overplanning_df.write \
.mode("overwrite") \
.format("delta") \
.save(material_shortage_raw_data_path)

# COMMAND ----------

spark.sql(f"DROP TABLE IF EXISTS {dbName}.material_shortage_raw")
spark.sql(f"CREATE TABLE {dbName}.material_shortage_raw USING DELTA LOCATION '{material_shortage_raw_data_path}'")

# COMMAND ----------

display(spark.sql(f"SELECT * FROM {dbName}.material_shortage_sku"))

# COMMAND ----------

display(spark.sql(f"SELECT * FROM {dbName}.material_shortage_raw"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Analyze the adjusted raw material demand
# MAGIC You can analyze the adjusted raw material demand using Databricks' simple dashboard functionality. See   [here](https://e2-demo-field-eng.cloud.databricks.com/sql/dashboards/fa660958-35a9-4710-a393-b050dd59275a-demand-analysis?edit&o=1444828305810485&p_w27bd4a0b-88a2-422a-bda5-9363bb3e7921_sku_parameter=%5B%22LRR_0X6CLF%22%5D&p_w6280d39b-f9b1-4644-b80c-caf98965b76e_sku_parameter=%5B%22LRR_0X6CLF%22%2C%22SRL_Z61857%22%5D).
