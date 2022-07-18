# Databricks notebook source
# MAGIC %md
# MAGIC # Manage a material shortage
# MAGIC After checking with the supplier how much raw material can actually be delivered we can now traverse the manufacturing value chain forwards to find out how much SKU's can actually be shipped to the customer. If one raw material is the bottleneck for producing a specific SKU, orders of the other raw materials for that SKU can be adjusted accordingly to save storage costs.

# COMMAND ----------

# MAGIC %md
# MAGIC *Prerequisite: Make sure to run 01_Simulate_Data, 03_Scaling_Model_Workflows and 04_Map_Demand_To_Raw before running this notebook.*
# MAGIC 
# MAGIC While the previous notebook *(04_Map_Demand_To_Raw)* demonstrated Databricks' graph functionality to traverse the manufacturing value chain backwards to find out how much raw material is needed for production, this notebook:
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

# MAGIC %md
# MAGIC ## Get all reported raw material shortages

# COMMAND ----------

# MAGIC %run ./Helper/Simulate_Material_Shortages

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from demand_db.material_shortage

# COMMAND ----------

# MAGIC %md
# MAGIC ## Get the affected SKU's
# MAGIC Get the affected SKU's and derive the quantity that can actually be shipped to the customer

# COMMAND ----------

demand_raw_df = spark.read.table("demand_db.forecast_raw")
material_shortage_df = spark.read.table("demand_db.material_shortage")

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

# Write the data 
affected_skus_df.write \
.mode("overwrite") \
.format("delta") \
.save('/FileStore/tables/demand_forecasting_solution_accelerator/material_shortage_sku/')

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE DATABASE IF NOT EXISTS demand_db;
# MAGIC DROP TABLE IF EXISTS demand_db.material_shortage_sku;
# MAGIC CREATE TABLE demand_db.material_shortage_sku USING DELTA LOCATION '/FileStore/tables/demand_forecasting_solution_accelerator/material_shortage_sku/'

# COMMAND ----------

# Write the data 
raw_overplanning_df.write \
.mode("overwrite") \
.format("delta") \
.save('/FileStore/tables/demand_forecasting_solution_accelerator/material_shortage_raw/')

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS demand_db.material_shortage_raw;
# MAGIC CREATE TABLE demand_db.material_shortage_raw USING DELTA LOCATION '/FileStore/tables/demand_forecasting_solution_accelerator/material_shortage_raw/'

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM demand_db.material_shortage_sku;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM demand_db.material_shortage_raw;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Analyze the adjusted raw material demand
# MAGIC You can analyze the adjusted raw material demand using Databricks' simple dashboard functionality. See   [here](https://e2-demo-field-eng.cloud.databricks.com/sql/dashboards/fa660958-35a9-4710-a393-b050dd59275a-demand-analysis?edit&o=1444828305810485&p_w27bd4a0b-88a2-422a-bda5-9363bb3e7921_sku_parameter=%5B%22LRR_0X6CLF%22%5D&p_w6280d39b-f9b1-4644-b80c-caf98965b76e_sku_parameter=%5B%22LRR_0X6CLF%22%2C%22SRL_Z61857%22%5D).
