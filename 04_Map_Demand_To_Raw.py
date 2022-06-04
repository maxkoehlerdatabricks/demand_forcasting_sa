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
                               ('Raw1', 'Intermediate1', 5),
                               ('Intermediate1','Intermediate2', 3),
                               ('Intermediate2', 'FinishedProduct', 1),
                               ('Raw2', 'Intermediate3', 5),
                               ('Intermediate3', 'FinishedProduct', 1),
                               ('FinishedProduct', 'SKU', 1) 
                              ],
                              ['src', 'dst', 'qty'])

# COMMAND ----------

./Pictures/typical_bom.png

# COMMAND ----------

# MAGIC %md
# MAGIC ![my_test_image](Pictures/typical_bom.png)

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://drive.google.com/file/d/1kujvsGKGb9LMpAbIcbbb08Z3FR4cni99/view?usp=sharing?raw=true" width=40%>

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://github.com/maxkoehlerdatabricks/demand_forcasting_sa/blob/max/Pictures/typical_bom.png ?raw=true" width=40%>

# COMMAND ----------


