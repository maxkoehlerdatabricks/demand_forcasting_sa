# Databricks notebook source
# MAGIC %md
# MAGIC # Map the forecasted demand to raw materials
# MAGIC Once the demand is forecasted, manufacturers need to purchase raw material and initiate production planning. This notebook shows how to translate future demand into raw materials. More precisely, we will do a Bill of Material (BoM) resolution to map the forecasted demand for each SKU to the appropriate demand of raw materials that are needed to produce the finished good that is mapped to the SKU.

# COMMAND ----------

import string
import networkx as nx
import random
import numpy as np
import pyspark.sql.functions as f
from graphframes import *
from graphframes.lib import *
AM = AggregateMessages
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
# MAGIC The above data frame represents a very simple BoM. It represents a building plan for a finished product. It consists of several intermediate products and raw materials. Quantities are also given. In reality a BoM consists of many more assembly steps and with an unknown number of assembly steps. Needless to say that this also means that there are many more raw materials and intermediate products. In this picture, we assume that the final product is mapped to one SKU. This information would not be part of a typical BoM. Note that a BoM is an information that is relevant mainly in production planning systems. An SKU would be something that is rather part of a logistics system. We assume that a look up table exists that maps each finished product to its SKU. The above BoM is then a result of artificially adding another assembly step with quantity 1. We now translate the manufacturing terms in terms that are used in graph theory. Each assembly step is an edge. The raw materials, intermediate products, the finished product and the SKU are vertices.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Prepration
# MAGIC The goal is to map the forecasted demand values for each SKU to quantities of the raw materials (the input of the production line) that are needed to produce the associated finished product (the output of the production line). To this end, we need a table which lists for each SKU demand for a time point all raw materials that are needed for production (ideally also at that time point to reduce warehouse costs). We do this in two steps:
# MAGIC Derive the SKU for each raw material.
# MAGIC Derive the product of all quantities of all succeeding assembly steps (=edges) from a raw material point of view.

# COMMAND ----------

# MAGIC %md
# MAGIC Derive all vertices

# COMMAND ----------

def create_vertices_from_edges(edges):
  vertices = ((edges.
   select(f.col('src')).
   distinct().
   withColumnRenamed('src','id')).
 union(
    (edges.
     select(f.col('dst')).
     distinct().
     withColumnRenamed('dst','id'))
 ).distinct()
 )
  return(vertices)

# COMMAND ----------

vertices = create_vertices_from_edges(edges)
display(vertices)

# COMMAND ----------

# MAGIC %md
# MAGIC Derive the graph

# COMMAND ----------

g = GraphFrame(vertices, edges)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1
# MAGIC The following function uses the concept of aggrgated messaging to derive a table that maps the raw for each SKU. 
# MAGIC 
# MAGIC See https://spark.apache.org/docs/latest/graphx-programming-guide.html

# COMMAND ----------

def get_sku_for_raw(gx):
  
  # Initiate Iteration
  iteration = 1
  
  # Inititate the edges
  updated_edges = gx.edges.select(f.col("src"),f.col("dst")).withColumn("aggregrated_parents", f.col("dst"))
 
  # Inititate the vertices
  updated_vertices = gx.vertices
  
  # Inititate the graph
  g_for_loop = GraphFrame(updated_vertices, updated_edges)
  
  # Initiate vertices_with_agg_messages
  emptyRDD = spark.sparkContext.emptyRDD()
  schema = StructType([
    StructField('id', StringType(), True),
    StructField('aggregrated_parents_from_parent', ArrayType(StringType(), True)),
    StructField('iteration', LongType(), True)
  ])
  vertices_with_agg_messages = spark.createDataFrame(emptyRDD,schema)
  
  
  while(True):
    
    ####THE WHILE LOOP STARTS HERE############################################################################
    
    #Aggregated Messaging
    msgToSrc = AM.edge["aggregrated_parents"]

    agg = g_for_loop.aggregateMessages(
     f.collect_set(AM.msg).alias("aggregrated_parents_from_parent"),
     sendToSrc=msgToSrc,
     sendToDst=None
    )

    agg = agg.withColumn("iteration", f.lit(iteration))

    if (iteration > 1):
      agg = agg.withColumn("aggregrated_parents_from_parent",f.flatten(f.col("aggregrated_parents_from_parent")))


    vertices_with_agg_messages = vertices_with_agg_messages.union(agg)
    
    #Increase iteration
    iteration+=1
    
    #Update edges
    updated_edges = g_for_loop.edges
    updated_edges = (updated_edges.
      join(agg, updated_edges["dst"] == agg["id"], how="inner").
      select(f.col("src"), f.col("dst"), f.col("aggregrated_parents_from_parent")).
      withColumnRenamed("aggregrated_parents_from_parent", "aggregrated_parents").
      withColumn("aggregrated_parents", f.array_union(f.col("aggregrated_parents"), f.array(f.col("dst")))).
      select(f.col("src"), f.col("dst"), f.col("aggregrated_parents"))
    )
    
    if (updated_edges.count() == 0):
      break
    
    #Update the graph
    g_for_loop = GraphFrame(updated_vertices, updated_edges)
    
    ####THE WHILE LOOP ENDS HERE#######################################################################
    
  # Subset to final iteration per id
  helper = (vertices_with_agg_messages.
    groupBy("id").
    agg(f.max("iteration").alias("iteration")))

  vertices_with_agg_messages = helper.join(vertices_with_agg_messages, ["id", "iteration"],  how="inner")

  # Subset to furthermost children
  in_degress_df = gx.inDegrees
  raw_df = (vertices.
   join( in_degress_df, ["id"], how='left_anti'))
  vertices_with_agg_messages = (raw_df.
                               join(vertices_with_agg_messages, ["id"],how="inner").select(f.col("id"),f.col("aggregrated_parents_from_parent"))
                              )
  vertices_with_agg_messages = (vertices_with_agg_messages.
                                 withColumn("SKU", f.col("aggregrated_parents_from_parent").getItem(0)).
                                 select(f.col("id"), f.col("SKU"))
                              )
    
    
  return(vertices_with_agg_messages)

# COMMAND ----------

res1 = get_sku_for_raw(g)
display(res1)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2
# MAGIC The following function uses the concept of aggrgated messaging to derive a table that maps the raw to the quantity that is needed to produce the desired finished product. For each raw material it is the product of quantities of all succeeding assemlby steps.
# MAGIC 
# MAGIC See https://spark.apache.org/docs/latest/graphx-programming-guide.html

# COMMAND ----------

def get_quantity_of_raw_needed_for_its_fin(gx):
  
  
  # Initialization:
  msgToSrc = AM.edge["qty"]
  
  # Initiate the graph to be updated by each loop iteration
  g_for_loop = gx
  
  # initiate vertices_with_agg_messages
  emptyRDD = spark.sparkContext.emptyRDD()
  schema = StructType([
  StructField('id', StringType(), True),
  StructField('qty', LongType(), True),
  StructField('iteration', LongType(), True)
  ])
  vertices_with_agg_messages = spark.createDataFrame(emptyRDD,schema)
  
  #Intita the iteration integer
  iteration = 1
  
  
  
  while(True):
    #Pass edge qty to child vertex
    agg = g_for_loop.aggregateMessages(
     f.first(AM.msg).alias("qty_from_parent"),
     sendToSrc=msgToSrc,
     sendToDst=None
    )
    
    #Update aggregation information table
    agg = agg.withColumn("iteration", f.lit(iteration))
    vertices_with_agg_messages = vertices_with_agg_messages.union(agg)
    
  
    #Update edges accordingly
    edges_old = g_for_loop.edges
    
    helper = (edges_old.
       join(agg, edges_old['dst'] == agg['id'], "left").
       filter(f.col("id").isNull()).
       select(f.col("src")).
       withColumnRenamed("src","to_multiply_look_up")
       )
    
    edges_update = edges_old.join(agg, edges_old['dst'] == agg['id'], "inner")
    edges_update = (edges_update.
           join(helper, edges_update["dst"] == helper["to_multiply_look_up"], "left").
                withColumn("qty", f.when(f.col("to_multiply_look_up").isNull(), f.col("qty")  ).otherwise(f.col("qty")*f.col("qty_from_parent"))).
                select(f.col('src'),f.col('dst'),f.col('qty'))
               )
    
    #Formulate Break condition
    if (edges_update.count()==0):
      break
      
    #Update iteration
    iteration+=1
    
    #Update Graph
    g_for_loop = GraphFrame(vertices, edges_update)
  
  
  #Subset to final iteration per id
  helper = (vertices_with_agg_messages.
    groupBy("id").
    agg(f.max("iteration").alias("iteration"))
         )

  vertices_with_agg_messages = helper.join(vertices_with_agg_messages, ["id", "iteration"],  how="inner")

  # Subset to furthermost children
  in_degress_df = g.inDegrees
  raw_df = (vertices.
   join( in_degress_df, ["id"], how='left_anti' )
  )
  vertices_with_agg_messages = raw_df.join(vertices_with_agg_messages, ["id"], how="inner").select(f.col("id"),f.col("qty"))
    
  #Return
  return(vertices_with_agg_messages)

# COMMAND ----------

res2 = get_quantity_of_raw_needed_for_its_fin(g)
display(res2)

# COMMAND ----------

# MAGIC %md
# MAGIC Joining the two tables yields the desired aggregated BoM

# COMMAND ----------

aggregated_bom = res1.join(res2, ["id"], how="inner").withColumnRenamed("id","RAW")
display(aggregated_bom)

# COMMAND ----------

# MAGIC %md
# MAGIC ## We now apply the concept on the forecasted demand data set

# COMMAND ----------

demand_df = spark.read.table("demand_db.part_level_demand")
sku_mapper = spark.read.table("demand_db.sku_mapper")
bom = spark.read.table("demand_db.bom")

# COMMAND ----------

display(demand_df)

# COMMAND ----------

# MAGIC %md
# MAGIC The BoM does not contain the mapping to SKU's. Threfore, we add an artifical assembly step with quantity 1

# COMMAND ----------

display(sku_mapper) 

# COMMAND ----------

# MAGIC %sql
# MAGIC select distinct SKU from demand_db.part_level_demand 

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

vertices = create_vertices_from_edges(edges)
display(vertices)

# COMMAND ----------

g = GraphFrame(vertices, edges)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1

# COMMAND ----------

res1 = get_sku_for_raw(g)
display(res1)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2

# COMMAND ----------

res2 = get_quantity_of_raw_needed_for_its_fin(g)
display(res2)

# COMMAND ----------

# MAGIC %md
# MAGIC Joining the two tables yields the desired aggregated BoM

# COMMAND ----------

aggregated_bom = res1.join(res2, ["id"], how="inner").withColumnRenamed("id","RAW").orderBy(f.col("SKU"),f.col("RAW"))
display(aggregated_bom)

# COMMAND ----------


