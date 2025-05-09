# Databricks notebook source
# MAGIC %md
# MAGIC # What is Databricks on SAP Business Data Cloud (BDC)?
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC `SAP Databricks`, is a customized offering that integrates Databricks' data science, SQL serverless, and AI/ML features into the BDC. It is particularly used by those customers that are transitioning their ERP and BW functionalities to the cloud under the "SAP RISE" initiative. A main component is that SAP provides managed data products stored in HANA Data Lake Files, which can then be shared to SAP Databricks or native Databricks using Delta Sharing. At the moment of writing this, only a few managed data products are available. However, there is a strong roadmap to extend.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://raw.githubusercontent.com/maxkoehlerdatabricks/demand_forcasting_sa/main/Pictures/SAP_BDC_Components.png" width="1500"/>

# COMMAND ----------

# MAGIC %md
# MAGIC # What is the Bill of Material?

# COMMAND ----------

# MAGIC %md
# MAGIC A Bill of Materials (BoM) table in SAP is a fundamental component used in various manufacturing and supply chain processes. It defines the components, quantities, and structure needed to produce a finished product. It lists all raw materials and parts required for the production of finished products. Each entry typically includes attributes such as material number, description, and quantity. BoMs can have multiple levels, indicating nested relationships where raw materials can consist of subassemblies. This hierarchical setup aids in understanding how individual components contribute to the final manufactured item.
# MAGIC
# MAGIC The BoM is utilized in various processes that include inventory management, production planning, and purchasing. In this article, it serves as a reference for calculating material requirements based on demand forecasts.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # How does the Bill of Material Table in this solution accelerator look like?
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC In our example the Bill of Material table consists of all raw materials that a final or intermediate material number consists of, along with related quantities. 
# MAGIC

# COMMAND ----------

dbutils.widgets.dropdown('reset_all_data', 'false', ['true', 'false'], 'Reset all data')
dbutils.widgets.text('catalogName',  'maxkoehler_demos' , 'Catalog Name')
dbutils.widgets.text('dbName',  'demand_db' , 'Database Name')

# COMMAND ----------

catalogName = dbutils.widgets.get('catalogName')
dbName = dbutils.widgets.get('dbName')
reset_all_data = dbutils.widgets.get('reset_all_data') == 'true'

# COMMAND ----------

bom_df = spark.read.table(f"{catalogName}.{dbName}.bom")
bom_df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC # Map the forecasted demand to raw materials
# MAGIC Traversing the manufacturing value chain backwards to find out how much raw material is needed to produce the forecasted number of products

# COMMAND ----------

# MAGIC %md
# MAGIC *Prerequisite: Make sure to run 01_Introduction_And_Setup and 02_Fine_Grained_Demand_Forecasting before running this notebook.*
# MAGIC
# MAGIC While the previous notebook *(002_Fine_Grained_Demand_Forecasting)* demonstrated the benefits of one of the Databricks' approach to train multiple models in parallel with great speed and cost-effectiveness,
# MAGIC in this part we show how to use Databricks' graph functionality to traverse the manufacturing value chain to find out how much raw material is needed for production.
# MAGIC
# MAGIC Key highlights for this notebook:
# MAGIC - Solve large scale graph problems by using GraphX as a distributed graph processing framework on top of Apache Spark
# MAGIC - Leverage the full support for property graphs to incorporate business knowlegde and the traverse the manufacturing value chain 

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://raw.githubusercontent.com/maxkoehlerdatabricks/demand_forcasting_sa/max/Pictures/operations_process_forwards.png" width="1500"/>

# COMMAND ----------

# MAGIC %md
# MAGIC Once the demand is forecasted, manufacturers need to purchase raw material and initiate production planning. This notebook shows how to translate future demand into raw materials. More precisely, we will do a Bill of Material (BoM) resolution to map the forecasted demand for each SKU to the appropriate demand of raw materials that are needed to produce the finished good that is mapped to the SKU.

# COMMAND ----------

#If True, all output files are in user specific databases, If False, a global database for the report is used
#user_based_data = True

# COMMAND ----------

#%run ./_resources_outside/00-global-setup $reset_all_data=false $db_prefix=demand_level_forecasting

# COMMAND ----------

#if (not user_based_data):
#  cloud_storage_path = '/FileStore/tables/demand_forecasting_solution_accelerator/'
#  dbName = 'demand_db' 
#  
##print(cloud_storage_path)
print(dbName)

# COMMAND ----------

# MAGIC %run ./_resources/00-setup $reset_all_data=false $catalogName=$catalogName $dbName=$dbName 

# COMMAND ----------

import os
import string
import random
import numpy as np
import pyspark.sql.functions as f
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, StringType, LongType

# COMMAND ----------

# MAGIC %md
# MAGIC ## We explain the algorithm first based on a simple example

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

# MAGIC %md
# MAGIC <img src="https://raw.githubusercontent.com/maxkoehlerdatabricks/demand_forcasting_sa/max/Pictures/typical_bom2.png" width="1500"/>

# COMMAND ----------

# MAGIC %md
# MAGIC The above data frame represents a very simple BoM. It represents a building plan for a finished product. It consists of several intermediate products and raw materials. Quantities are also given. In reality, a BoM consists of many more and previously unknown number of steps. Needless to say that this also means that there are many more raw materials and intermediate products. In this picture, we assume that the final product is mapped to one SKU. This information would not be part of a typical BoM. Note that a BoM is mainly relevant in production planning systems, whereas an SKU would be something that is rather part of a logistics system. We assume that a look up table exists that maps each finished product to its SKU. The above BoM is then a result of artificially adding another step with quantity 1. We now translate the manufacturing terms in terms that are used in graph theory: Each assembly step is an edge; the raw materials, intermediate products, the finished product and the SKU are vertices.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Prepration
# MAGIC The goal is to map the forecasted demand values for each SKU to quantities of the raw materials (the input of the production line) that are needed to produce the associated finished product (the output of the production line). To this end, we need a table which lists for each SKU demand for a time point all raw materials that are needed for production (ideally also at that time point to reduce warehouse costs). We do this in two steps:
# MAGIC - Step 1: Derive the SKU for each raw material.
# MAGIC - Step 2: Derive the product of all quantities of all succeeding assembly steps (=edges) from a raw material point of view.

# COMMAND ----------

def get_skus(edges):
  skus_df = edges.alias("e1").join(edges.alias("e2").select("src").distinct(), f.col("e1.dst") == f.col("e2.src"), "left_anti").selectExpr("dst as component", "qty as total_qty", "dst as sku").distinct() 
  return skus_df              

# COMMAND ----------

skus_df = get_skus(edges)

display(skus_df)

# COMMAND ----------

def get_raw_materials(edges):
  raw_df = edges.alias("e1").select("src").join(edges.alias("e2").select("dst").distinct(), f.col("e1.src") == f.col("e2.dst"), "left_anti").distinct()
  return raw_df


# COMMAND ----------

raw_df = get_raw_materials(edges)
display(raw_df)

# COMMAND ----------

def get_raws_for_skus(skus_df, edges, raw_df):

  bom_traversal_df = skus_df

  while True:
    bom_traversal_df = bom_traversal_df.alias("bom").join(edges.alias("edges"), bom_traversal_df.component == edges.dst, "left").selectExpr("edges.src as component", "bom.total_qty * edges.qty as total_qty", "sku")

    if bom_traversal_df.where(f.col("component").isNotNull()).count() == 0:
      break

    skus_df = skus_df.union(bom_traversal_df)
  # filter skus_df to only include raw materials
  raw_marterials = [row.src for row in raw_df.select("src").collect()]
  result_df = skus_df.filter(f.col("component").isin(raw_marterials)).select(f.col("sku").alias("SKU"), f.col("component").alias("RAW"), f.col("total_qty").alias("QTY_RAW")).groupBy("SKU", "RAW").agg(f.sum("QTY_RAW").alias("QTY_RAW")).orderBy("SKU",f.col("QTY_RAW").desc())
  return result_df

# COMMAND ----------

result_df = get_raws_for_skus(skus_df, edges, raw_df)
display(result_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## We now apply the concept on the forecasted demand data set

# COMMAND ----------

demand_df = spark.read.table(f"{catalogName}.{dbName}.part_level_demand_with_forecasts")
sku_mapper = spark.read.table(f"{catalogName}.{dbName}.sku_mapper")
bom = spark.read.table(f"{catalogName}.{dbName}.bom")

# COMMAND ----------

demand_df = (demand_df.
        withColumn("Demand", f.col("Demand_Fitted")).
        select(f.col("Product"), f.col("SKU"), f.col("Date"), f.col("Demand")))

# COMMAND ----------

display(demand_df)

# COMMAND ----------

# MAGIC %md
# MAGIC The BoM does not contain the mapping to SKU's. Threfore, we add an artifical assembly step with quantity 1

# COMMAND ----------

display(bom)

# COMMAND ----------

display(sku_mapper) 

# COMMAND ----------

display(spark.sql(f"select distinct SKU from part_level_demand_with_forecasts"))

# COMMAND ----------

edges = (sku_mapper.withColumn("qty", f.lit(1)).
  withColumnRenamed("final_mat_number", "material_in").
  withColumnRenamed("sku","material_out").
  union(bom).
  withColumnRenamed("material_in","src").
  withColumnRenamed("material_out","dst")
        )
display(edges)       

# COMMAND ----------

skus_df = get_skus(edges)
raw_df = get_raw_materials(edges)

# COMMAND ----------

result_df = get_raws_for_skus(skus_df, edges, raw_df)

# COMMAND ----------

display(result_df)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Derive the demand for raw material

# COMMAND ----------

demand_raw_df = (demand_df.
      join(result_df, ["SKU"], how="inner").
      select("Product","SKU","RAW", "Date","Demand", "QTY_RAW").
      withColumn("Demand_Raw", f.col("QTY_RAW")*f.col("Demand")).
      withColumnRenamed("Demand","Demand_SKU").
      orderBy(f.col("SKU"),f.col("RAW"), f.col("Date"))
                )
display(demand_raw_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save to delta

# COMMAND ----------

demand_raw_df.write.mode("overwrite").saveAsTable(f"{catalogName}.{dbName}.forecast_raw")
