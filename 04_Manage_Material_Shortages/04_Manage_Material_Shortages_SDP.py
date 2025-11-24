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

#dbutils.widgets.dropdown('reset_all_data', 'false', ['true', 'false'], 'Reset all data')
#dbutils.widgets.text('catalogName',  'maxkoehler_demos' , 'Catalog Name')
#dbutils.widgets.text('dbName',  'demand_db' , 'Database Name')

# COMMAND ----------

#catalogName = dbutils.widgets.get('catalogName')
#dbName = dbutils.widgets.get('dbName')
#reset_all_data = dbutils.widgets.get('reset_all_data') == 'true'

# COMMAND ----------

# %run ./_resources/00-setup $reset_all_data=false $catalogName=$catalogName $dbName=$dbName 

# COMMAND ----------

import dlt
import os
import random
import pyspark.sql.functions as f
from pyspark.sql.types import FloatType

# COMMAND ----------

# MAGIC %md
# MAGIC ### Building a Spark Declarative Pipeline to analyze and manage raw material shortages
# MAGIC
# MAGIC In this example we'll implement an end-2-end spark declarative pipeline consuming our forecasted raw parts demand. 
# MAGIC
# MAGIC We'll incrementally load new data and simulate raw material shortages, which we'll then use to find affected skus and adjust our production quantities according to what can be shipped to our customers. 
# MAGIC
# MAGIC This information will then be used to:
# MAGIC   - Build our AI/BI Dashboard to track material shortages and their impact
# MAGIC   - Set-up a Genie Space to provide Analytics Self-Service and enable natural language interactions for Ad-Hoc questions

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://raw.githubusercontent.com/maxkoehlerdatabricks/demand_forcasting_sa/main/Pictures/dlt_pipeline_graph.png" width="1500"/>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Get all reported raw material shortages

# COMMAND ----------

@dlt.table(name="get_material_shortages", comment="All reported material shortages for raw materials")
def get_material_shortages():
    demand_raw_df = spark.read.table("forecast_raw")
    material_shortages_sku = [row['SKU'] for row in demand_raw_df.select('SKU').distinct().orderBy(f.rand()).limit(2).collect()]
    material_shortages_raw = [row['RAW'] for row in demand_raw_df.filter(f.col("SKU").isin(material_shortages_sku)).select('RAW').distinct().orderBy(f.rand()).limit(3).collect()]
    maximum_date = demand_raw_df.agg(f.max("Date").alias("max_date")).collect()[0]['max_date']
    return (
        demand_raw_df
        .filter(
            (f.col("SKU").isin(material_shortages_sku)) &
            (f.col("Date") == maximum_date) &
            (f.col("RAW").isin(material_shortages_raw))
        )
        .withColumn("fraction", (f.rand() * 0.4) + 0.5)
        .withColumn("available_demand", f.floor(f.col("fraction") * f.col("Demand_Raw")))
        .select("RAW", "Date", "available_demand")
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Get the affected SKU's
# MAGIC Get the affected SKU's and derive the quantity that can actually be shipped to the customer

# COMMAND ----------

@dlt.table(name="affected_skus", comment="SKUs affected by material shortages")
def affected_skus():
  material_shortage_df = spark.read.table("get_material_shortages")
  demand_raw_df = spark.read.table("forecast_raw")
  return (material_shortage_df
            .join(demand_raw_df, ["RAW", "Date"], how="inner")
            .withColumn("fraction_raw", f.col("available_demand") / f.col("Demand_Raw"))
          )

@dlt.table(name="min_fraction", comment="Minimum fraction of demand for raw material for affected SKUs")
def min_fraction():
  affected_skus_df = spark.read.table("affected_skus")                    
  return (affected_skus_df
            .groupBy("SKU")
            .agg(f.min(f.col("fraction_raw")).alias("min_fraction_raw"))
          )

@dlt.table(name="demand_adjusted", comment="Demand adjusted for raw material shortages")
def demand_adjusted():
  affected_skus_df = spark.read.table("affected_skus")
  min_fraction = spark.read.table("min_fraction")
  return (affected_skus_df
          .join(min_fraction, ["SKU"])
          .withColumnRenamed("available_demand", "Adjusted_Demand_Raw")
          .withColumn("Adjusted_Demand_SKU", f.floor(f.col("Demand_SKU") * f.col("min_fraction_raw")))
          .select(
            f.col("RAW"), 
            f.col("Date"), 
            f.col("SKU").alias("Affected_SKU"), 
            f.col("Product").alias("Affected_Product"), 
            f.col("Demand_RAW"), 
            f.col("Adjusted_Demand_Raw"), 
            f.col("Demand_SKU"), 
            f.col("Adjusted_Demand_SKU"), 
            f.col("min_fraction_raw").alias("Available_Fraction_For_SKU")
          )
         )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Get the amount of overplanning other raw materials
# MAGIC If one raw material has a shortage, other raw materials in the specific SKU are overplanned and orders can be adjusted to reduce storage costs

# COMMAND ----------

# MAGIC %md
# MAGIC We're adding an expectation on different fields to enforce and track our Data Quality.<br>
# MAGIC This will ensure that our data is valid and easily spot potential errors due to data anomaly.

# COMMAND ----------

@dlt.table(name="raw_overplanning", comment="Raw material adjusted to prevent overplanning due to material shortages")
@dlt.expect_or_drop("valid raw", "RAW IS NOT NULL")
@dlt.expect_or_drop("valid demand raw adjusted", "Demand_Raw_Adjusted >= 0")
@dlt.expect_or_drop("valid date", "Date IS NOT NULL AND Date <= current_date() + interval 90 days")
def raw_overplanning():
  affected_skus_df = spark.read.table("demand_adjusted")
  demand_raw_df = spark.read.table("forecast_raw")
  return (affected_skus_df
            .select(f.col("Affected_SKU").alias("SKU"), f.col("Date"), f.col("Available_Fraction_For_SKU"))
            .join(demand_raw_df, ["SKU", "Date"], how="inner")
            .withColumn("Demand_Raw_Adjusted", f.floor(f.col("Demand_RAW") * f.col("Available_Fraction_For_SKU")))
            .select(f.col("RAW"), f.col("Date"), f.col("Demand_Raw"), f.col("Demand_Raw_Adjusted"))
          )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Analyze the adjusted raw material demand
# MAGIC You can analyze the adjusted raw material demand using Databricks' simple dashboard functionality. See   [here](https://e2-demo-field-eng.cloud.databricks.com/sql/dashboards/fa660958-35a9-4710-a393-b050dd59275a-demand-analysis?edit&o=1444828305810485&p_w27bd4a0b-88a2-422a-bda5-9363bb3e7921_sku_parameter=%5B%22LRR_0X6CLF%22%5D&p_w6280d39b-f9b1-4644-b80c-caf98965b76e_sku_parameter=%5B%22LRR_0X6CLF%22%2C%22SRL_Z61857%22%5D).
