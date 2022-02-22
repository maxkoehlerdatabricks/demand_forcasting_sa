# Databricks notebook source
# MAGIC %md
# MAGIC References:
# MAGIC * https://databricks.com/notebooks/simple-aws/petastorm-spark-converter-pytorch.html
# MAGIC * https://databricks.com/notebooks/simple-aws/petastorm-spark-converter-tensorflow.html
# MAGIC * https://docs.microsoft.com/en-us/learn/modules/deep-learning-with-horovod-distributed-training/

# COMMAND ----------

# Read in data
data_path = '/FileStore/tables/demand_forecasting_solution_accellerator/demand_df_delta/'
demand_df = spark \
  .read \
  .format("delta") \
  .load(data_path)

display(demand_df)
