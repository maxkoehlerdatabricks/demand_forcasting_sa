# Databricks notebook source
# MAGIC %md
# MAGIC # Manage a supplier shortage
# MAGIC After checking with the supplier how much raw material can actually be delivered we can now traversing the manufacturing value chain forwards to find out how much SKU's can actually be shipped to the customer. If one raw material is the bottleneck for producing a specific SKU, orders of the other raw materials for that SKU can be adjusted accordingly to save storage costs.

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
# MAGIC # Get raw material shortages

# COMMAND ----------

# MAGIC %run ./Helper/Simulate_Material_Shortages

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from demand_db.material_shortage

# COMMAND ----------


