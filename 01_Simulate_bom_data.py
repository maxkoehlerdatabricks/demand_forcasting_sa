# Databricks notebook source
# MAGIC %md
# MAGIC # Simulate Bill of Material (BoM) Data
# MAGIC This notebook simulates typical BoM data for a set of SKU's

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC *Prerequisite: Make sure to run the data simulation notebook 01_Simulate_Data before running this notebook.*

# COMMAND ----------

import string
import networkx as nx
import random
import numpy as np

# COMMAND ----------

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def generate_random_strings(n):
  random.seed(123)
  random_mat_numbers = set()
  while True:
    random_mat_numbers.add(id_generator(size=5))
    if len(random_mat_numbers) >= n:
      break
  return(random_mat_numbers)

# COMMAND ----------

def extend_one_step(node_from_):
  res_ = [  ]
  node_list_to_be_extended_ = [  ]
  # second level
  random_split_number = random.randint(2, 4)
  for i in range(random_split_number):
    node_to = random_mat_numbers.pop()
    node_list_to_be_extended_.append(node_to)
    res_.append((node_from_, node_to))
  return res_, node_list_to_be_extended_

# COMMAND ----------

def extend_one_level(node_list_to_be_extended, level, sku):
  
  
  print(f"""In  'extend_one_level': level={level} and sku = {sku}  """)
  
  if level == 1:
    head_node = random_mat_numbers.pop() 
    node_list_to_be_extended_one_level = [ ]
    node_list_to_be_extended_one_level.append(head_node)
    res_one_level = [ (head_node, sku) ]
  else:
    res_one_level = [ ]
    node_list_to_be_extended_one_level = [ ]
    
    if len(node_list_to_be_extended) > 2:
      node_list_to_be_extended_ = node_list_to_be_extended[ : 3 ]
    else:
      node_list_to_be_extended_ = node_list_to_be_extended

    for node in node_list_to_be_extended_:
      res_one_step = [ ]
      node_list_to_be_extended_one_step = [ ]
      
      res_one_step, node_list_to_be_extended_one_step = extend_one_step(node)    
      res_one_level.extend(res_one_step)
      node_list_to_be_extended_one_level.extend(node_list_to_be_extended_one_step)
  
  return res_one_level, node_list_to_be_extended_one_level

# COMMAND ----------

#Generate a set of material numbers
random_mat_numbers = generate_random_strings(1000000)

# COMMAND ----------

#Create a listof all SKU's
demand_df = spark.read.table("demand_db.part_level_demand")
all_skus = demand_df.select('SKU').distinct().rdd.flatMap(lambda x: x).collect()

# COMMAND ----------

# Generaze edges
depth = 3
new_node_list = [ ]
edge_list = [ ]

for sku in all_skus:
  for level_ in range(1, (depth + 1)):
    new_edge_list, new_node_list = extend_one_level(new_node_list, level = level_, sku=sku)
    edge_list.extend(new_edge_list)

# COMMAND ----------

# Define the graph 
g=nx.Graph()
g.add_edges_from(edge_list)  

# COMMAND ----------

# Assign a quantity for the graph
edge_df = nx.to_pandas_edgelist(g)
edge_df = edge_df.assign(qty = np.where(edge_df.target.str.len() == 10, 1, np.random.randint(1,4, size=edge_df.shape[0])))

# COMMAND ----------

#Create the fnal mat number to sku mapper
final_mat_number_to_sku_mapper = edge_df[edge_df.target.str.match('SRL_.*')][["source","target"]]
final_mat_number_to_sku_mapper = final_mat_number_to_sku_mapper.rename(columns={"source": "final_mat_number", "target": "sku"} )

# COMMAND ----------

# Create BoM
bom = edge_df[~edge_df.target.str.match('SRL_.*')]
bom = bom.rename(columns={"source": "material_in", "target": "material_out"} )

# COMMAND ----------

bom_df = spark.createDataFrame(bom) 
final_mat_number_to_sku_mapper_df = spark.createDataFrame(final_mat_number_to_sku_mapper)

# COMMAND ----------

# Write the data 
bom_df.write \
.mode("overwrite") \
.format("delta") \
.save('/FileStore/tables/demand_forecasting_solution_accelerator/bom_df_delta/')

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE DATABASE IF NOT EXISTS demand_db;
# MAGIC DROP TABLE IF EXISTS demand_db.bom;
# MAGIC CREATE TABLE demand_db.bom USING DELTA LOCATION '/FileStore/tables/demand_forecasting_solution_accelerator/bom_df_delta/'

# COMMAND ----------

final_mat_number_to_sku_mapper_df.write \
.mode("overwrite") \
.format("delta") \
.save('/FileStore/tables/demand_forecasting_solution_accelerator/sku_mapper_df_delta/')

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS demand_db.sku_mapper;
# MAGIC CREATE TABLE demand_db.sku_mapper USING DELTA LOCATION '/FileStore/tables/demand_forecasting_solution_accelerator/sku_mapper_df_delta/'

# COMMAND ----------


