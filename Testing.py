# Databricks notebook source
# MAGIC %md 
# MAGIC # Testing notebook
# MAGIC bala.amavasai@databrics.com

# COMMAND ----------

# MAGIC %sql
# MAGIC USE demand_db;
# MAGIC SHOW TABLES;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM demand_df_delta;
# MAGIC ;SELECT * FROM hierarchical_ts_table;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM demand_df_delta WHERE SKU='CAM_1MIEJZ'

# COMMAND ----------


