# Databricks notebook source
# MAGIC %md
# MAGIC # Map the forecasted demand to raw materials
# MAGIC Once the demand is forecasted, manufacturers need to purchase raw material and initiate production planning. This notebook shows how to do a Bill of Material (BoM) resolution to map the forecasted demand of the raw materials to the appropriate demand of raw materials.

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### We explain the algorithm first and then apply it the table with the forecasted demand for each SKU

# COMMAND ----------

# Let's create an easy BoM data set
edges = spark.createDataFrame([
                               ('v1', 'v2', 2),
                               ('v2', 'v3', 3),
                               ('v3', 'v4', 4),
                               ('v5', 'v4', 6),
                               ('v6', 'v5', 7)
                              ],
                              ['src', 'dst', 'qty'])

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------


