# Databricks notebook source
# MAGIC %md
# MAGIC # Look at data

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM demand_db.part_level_demand_with_forecasts

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT distinct SKU, product FROM demand_db.part_level_demand_with_forecasts

# COMMAND ----------

# MAGIC %md
# MAGIC # Some tries with Graphs

# COMMAND ----------

# MAGIC %pip install igraph
# MAGIC %pip install cairocffi
# MAGIC %pip install --upgrade networkx
# MAGIC %pip install --upgrade pydot

# COMMAND ----------

from igraph import *
import cairocffi
import matplotlib.pyplot as plt
import networkx as nx
import random

# COMMAND ----------

# Sample graph
g = Graph.Tree(n=30, children=5)

layout = g.layout("kk")
fig, ax = plt.subplots()
plot(g, layout=layout, target=ax)

# COMMAND ----------

A = g.get_edgelist()
g_nx = nx.DiGraph(A) # In case your graph is directed

# COMMAND ----------

#nx.to_pandas_dataframe(g_nx)
nx.to_pandas_edgelist(g_nx)

# COMMAND ----------

G = nx.complete_graph(5)
nx.draw(G)

# COMMAND ----------

def hierarchy_pos(G, root=None, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5):

    '''
    From Joel's answer at https://stackoverflow.com/a/29597209/2966723.  
    Licensed under Creative Commons Attribution-Share Alike 
    
    If the graph is a tree this will return the positions to plot this in a 
    hierarchical layout.
    
    G: the graph (must be a tree)
    
    root: the root node of current branch 
    - if the tree is directed and this is not given, 
      the root will be found and used
    - if the tree is directed and this is given, then 
      the positions will be just for the descendants of this node.
    - if the tree is undirected and not given, 
      then a random choice will be used.
    
    width: horizontal space allocated for this branch - avoids overlap with other branches
    
    vert_gap: gap between levels of hierarchy
    
    vert_loc: vertical location of root
    
    xcenter: horizontal location of root
    '''
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  #allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5, pos = None, parent = None):
        '''
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        '''
    
        if pos is None:
            pos = {root:(xcenter,vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)  
        if len(children)!=0:
            dx = width/len(children) 
            nextx = xcenter - width/2 - dx/2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G,child, width = dx, vert_gap = vert_gap, 
                                    vert_loc = vert_loc-vert_gap, xcenter=nextx,
                                    pos=pos, parent = root)
        return pos

            
    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)

# COMMAND ----------

import matplotlib.pyplot as plt
import networkx as nx
G=nx.Graph()
G.add_edges_from([(1,2), (1,3), (1,4), (2,5), (2,6), (2,7), (3,8), (3,9), (4,10),
                  (5,11), (5,12), (6,13)])
pos = hierarchy_pos(G,1)    
nx.draw(G, pos=pos, with_labels=True)

# COMMAND ----------

random_depth = random.randint(3, 3)
random_split_number = random.randint(3, 5)

# COMMAND ----------

# MAGIC %md
# MAGIC # Develop an algorith to simulate the BOM data

# COMMAND ----------

#Libraries
import string
import networkx as nx
import random
import numpy as np
import pyspark.sql.functions as f
from graphframes import *

# COMMAND ----------

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

# Create a Pandas UDF to simulate unique SKU's, i.e. n random strings without repetition
def id_sequence_generator(pdf):
  random.seed(123)
  res = set()
  while True:
    res.add(id_generator())
    if len(res) >= n:
      break

# COMMAND ----------

def generate_random_strings(n):
  random.seed(123)
  random_mat_numbers = set()
  while True:
    random_mat_numbers.add(id_generator(size=5))
    if len(random_mat_numbers) >= n:
      break
  return(random_mat_numbers)

# COMMAND ----------

random_mat_numbers = generate_random_strings(1000)

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

new_node_list = [ ]
edge_list, new_node_list = extend_one_level(new_node_list, level = 1, sku="HEAD_SKU")

# COMMAND ----------

new_node_list

# COMMAND ----------

edge_list

# COMMAND ----------

edge_list2, new_node_list2 = extend_one_level(new_node_list, level = 2, sku="HEAD_SKU")

# COMMAND ----------

edge_list2

# COMMAND ----------

new_node_list2

# COMMAND ----------

edge_list3, new_node_list3 = extend_one_level(new_node_list2, level = 3, sku="HEAD_SKU")

# COMMAND ----------

edge_list3

# COMMAND ----------

new_node_list3

# COMMAND ----------

edge_list4, new_node_list4 = extend_one_level(new_node_list3, level = 4, sku="HEAD_SKU")

# COMMAND ----------

edge_list4

# COMMAND ----------

new_node_list4

# COMMAND ----------

## Loop
depth = 5
new_node_list = [ ]
edge_list = [ ]
for level_ in range(1, (depth + 1)):
  new_edge_list, new_node_list = extend_one_level(new_node_list, level = level_, sku="HEAD_SKU")
  edge_list.extend(new_edge_list)

# COMMAND ----------

G=nx.Graph()
G.add_edges_from(edge_list)
pos = hierarchy_pos(G, edge_list[0][0]  )    
nx.draw(G, pos=pos, with_labels=False)

# COMMAND ----------

nx.draw(G, with_labels=True)

# COMMAND ----------

edge_list

# COMMAND ----------

# MAGIC %md
# MAGIC Loop over all SKU's

# COMMAND ----------

random_mat_numbers = generate_random_strings(1000000)

# COMMAND ----------

len(random_mat_numbers)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT distinct SKU FROM demand_db.part_level_demand_with_forecasts

# COMMAND ----------

demand_df = spark.read.table("demand_db.part_level_demand")
display(demand_df)

# COMMAND ----------

all_skus = demand_df.select('SKU').distinct().rdd.flatMap(lambda x: x).collect()

# COMMAND ----------

type(all_skus)

# COMMAND ----------

####LOOP!!!!!!!
depth = 5
new_node_list = [ ]
edge_list = [ ]

for sku in all_skus:
  for level_ in range(1, (depth + 1)):
    new_edge_list, new_node_list = extend_one_level(new_node_list, level = level_, sku=sku)
    edge_list.extend(new_edge_list)

# COMMAND ----------

G=nx.Graph()
G.add_edges_from(edge_list)   
#nx.draw(G, with_labels=False)

# COMMAND ----------

edge_df = nx.to_pandas_edgelist(G)

# COMMAND ----------

edge_df = edge_df.assign(qty = np.where(edge_df.target.str.len() == 10, 1, np.random.randint(1,4, size=edge_df.shape[0])))

# COMMAND ----------

edge_df

# COMMAND ----------

edges =spark.createDataFrame(edge_df) 
edges.printSchema()
edges.show()

# COMMAND ----------

edges = (edges.
  withColumnRenamed('source','src').
  withColumnRenamed('target','dst'))

# COMMAND ----------

vertices = ((edges.
   select(f.col('src')).
   distinct().
   withColumnRenamed('src','id')).
 union(
    (edges.
     select(f.col('dst')).
     distinct().
     withColumnRenamed('dst','id'))
 )
 )

# COMMAND ----------

g = GraphFrame(vertices, edges)

# COMMAND ----------

display(g.inDegrees.filter('inDegree == 0'))

# COMMAND ----------

display(g.outDegrees)

# COMMAND ----------

in_degress_df = g.inDegrees
raw_df = (vertices.
   join( in_degress_df, [  vertices.id == in_degress_df.id ], how='left_anti' )
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Create a graph to develop the following algorithms

# COMMAND ----------

edges = spark.createDataFrame([
                               ('v1', 'v2', 2),
                               ('v2', 'v3', 3),
                               ('v3', 'v4', 4),
                               ('v5', 'v4', 6),
                               ('v6', 'v5', 7)
                              ],
                              ['src', 'dst', 'qty'])

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

# COMMAND ----------

vertices.show()

# COMMAND ----------

g = GraphFrame(vertices, edges)

# COMMAND ----------

nx.draw(
  nx.from_pandas_edgelist(g.edges.toPandas().rename(columns={"src": "source", "dst": "target"})),
  with_labels = True
)

# COMMAND ----------

g.vertices.show()

# COMMAND ----------

g.edges.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Libraries

# COMMAND ----------

from graphframes.lib import *
from graphframes import GraphFrame
AM = AggregateMessages
import pyspark.sql.functions as f

# COMMAND ----------

# MAGIC %md
# MAGIC #Develop an algorithm to deduce a table with the raw materials that are in each SKU

# COMMAND ----------

#Initiate Iteration
iteration = 1

# COMMAND ----------

# Inititate the edges
updated_edges = g.edges.select(f.col("src"),f.col("dst")).withColumn("aggregrated_parents", f.col("dst"))
updated_edges.show()

# COMMAND ----------

# Inititate the vertices
updated_vertices = g.vertices
updated_vertices.show()

# COMMAND ----------

# Inititate the graph
g_for_loop = GraphFrame(updated_vertices, updated_edges)

# COMMAND ----------

# initiate vertices_with_agg_essages
emptyRDD = spark.sparkContext.emptyRDD()
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, StringType, LongType
schema = StructType([
  StructField('id', StringType(), True),
  StructField('aggregrated_parents_from_parent', ArrayType(StringType(), True)),
  StructField('iteration', LongType(), True)
  ])
vertices_with_agg_essages = spark.createDataFrame(emptyRDD,schema)

vertices_with_agg_essages.show()

# COMMAND ----------

#Let's try the first value
msgToSrc = AM.edge["aggregrated_parents"]

agg = g_for_loop.aggregateMessages(
 f.collect_set(AM.msg).alias("aggregrated_parents_from_parent"),
 sendToSrc=msgToSrc,
 sendToDst=None
)

agg = agg.withColumn("iteration", f.lit(iteration))

if (iteration > 1):
  agg = agg.withColumn("aggregrated_parents_from_parent",f.flatten(f.col("aggregrated_parents_from_parent")))

  
vertices_with_agg_essages = vertices_with_agg_essages.union(agg)

vertices_with_agg_essages.show()

# COMMAND ----------

iteration+=1

# COMMAND ----------

#Update edges
updated_edges = g_for_loop.edges
updated_edges = (updated_edges.
  join(agg, updated_edges["dst"] == agg["id"], how="inner").
  select(f.col("src"), f.col("dst"), f.col("aggregrated_parents_from_parent")).
  withColumnRenamed("aggregrated_parents_from_parent", "aggregrated_parents").
  withColumn("aggregrated_parents", f.array_union(f.col("aggregrated_parents"), f.array(f.col("dst")))).
  select(f.col("src"), f.col("dst"), f.col("aggregrated_parents"))
)

updated_edges.show()

# COMMAND ----------

#Update Vertices
#updated_vertices = create_vertices_from_edges(updated_edges)
#updated_vertices.show()

# COMMAND ----------

#Update the graph
g_for_loop = GraphFrame(updated_vertices, updated_edges)

# COMMAND ----------

#Let's try the second value
msgToSrc = AM.edge["aggregrated_parents"]

agg = g_for_loop.aggregateMessages(
 f.collect_set(AM.msg).alias("aggregrated_parents_from_parent"),
 sendToSrc=msgToSrc,
 sendToDst=None
)

agg = agg.withColumn("iteration", f.lit(iteration))

if (iteration > 1):
  agg = agg.withColumn("aggregrated_parents_from_parent",f.flatten(f.col("aggregrated_parents_from_parent")))

vertices_with_agg_essages = vertices_with_agg_essages.union(agg)

vertices_with_agg_essages.show()

# COMMAND ----------

#Update edges
updated_edges = g_for_loop.edges
updated_edges = (updated_edges.
  join(agg, updated_edges["dst"] == agg["id"], how="inner").
  select(f.col("src"), f.col("dst"), f.col("aggregrated_parents_from_parent")).
  withColumnRenamed("aggregrated_parents_from_parent", "aggregrated_parents").
  withColumn("aggregrated_parents", f.array_union(f.col("aggregrated_parents"), f.array(f.col("dst")))).
  select(f.col("src"), f.col("dst"), f.col("aggregrated_parents"))
)

updated_edges.show()

# COMMAND ----------

iteration+=1

# COMMAND ----------

#Update edges
updated_edges = g_for_loop.edges
updated_edges = (updated_edges.
  join(agg, updated_edges["dst"] == agg["id"], how="inner").
  select(f.col("src"), f.col("dst"), f.col("aggregrated_parents_from_parent")).
  withColumnRenamed("aggregrated_parents_from_parent", "aggregrated_parents").
  withColumn("aggregrated_parents", f.array_union(f.col("aggregrated_parents"), f.array(f.col("dst")))).
  select(f.col("src"), f.col("dst"), f.col("aggregrated_parents"))
)

updated_edges.show()

# COMMAND ----------

#Update Vertices
#updated_vertices = create_vertices_from_edges(updated_edges)
#updated_vertices.show()

# COMMAND ----------

#Update the graph
g_for_loop = GraphFrame(updated_vertices, updated_edges)

# COMMAND ----------

#Let's try the third value
msgToSrc = AM.edge["aggregrated_parents"]

agg = g_for_loop.aggregateMessages(
 f.collect_set(AM.msg).alias("aggregrated_parents_from_parent"),
 sendToSrc=msgToSrc,
 sendToDst=None
)

agg = agg.withColumn("iteration", f.lit(iteration))

if (iteration > 1):
  agg = agg.withColumn("aggregrated_parents_from_parent",f.flatten(f.col("aggregrated_parents_from_parent")))

vertices_with_agg_essages = vertices_with_agg_essages.union(agg)

vertices_with_agg_essages.show()

# COMMAND ----------

iteration+=1

# COMMAND ----------

#Update edges
updated_edges = g_for_loop.edges
updated_edges = (updated_edges.
  join(agg, updated_edges["dst"] == agg["id"], how="inner").
  select(f.col("src"), f.col("dst"), f.col("aggregrated_parents_from_parent")).
  withColumnRenamed("aggregrated_parents_from_parent", "aggregrated_parents").
  withColumn("aggregrated_parents", f.array_union(f.col("aggregrated_parents"), f.array(f.col("dst")))).
  select(f.col("src"), f.col("dst"), f.col("aggregrated_parents"))
)

updated_edges.show()

# COMMAND ----------

#if (updated_edges.count() == 0):
#  break

# COMMAND ----------

#Result

#Subset to final iteration per id
helper = (vertices_with_agg_essages.
  groupBy("id").
  agg(f.max("iteration").alias("iteration"))
       )

vertices_with_agg_essages = helper.join(vertices_with_agg_essages, ["id", "iteration"],  how="inner")

# Subset to furthermost children
in_degress_df = g.inDegrees
raw_df = (vertices.
 join( in_degress_df, ["id"], how='left_anti' )
)
vertices_with_agg_essages = raw_df.join(vertices_with_agg_essages, ["id"], how="inner").select(f.col("id"),f.col("aggregrated_parents_from_parent"))
vertices_with_agg_essages = vertices_with_agg_essages.withColumn("SKU", f.col("aggregrated_parents_from_parent").getItem(0)).select(f.col("id"), f.col("SKU"))

vertices_with_agg_essages.show()

# COMMAND ----------

### Now loop
def get_sku_for_raw(gx):
  
  # Initiate Iteration
  iteration = 1
  
  # Inititate the edges
  updated_edges = gx.edges.select(f.col("src"),f.col("dst")).withColumn("aggregrated_parents", f.col("dst"))
 
  # Inititate the vertices
  updated_vertices = gx.vertices
  
  # Inititate the graph
  g_for_loop = GraphFrame(updated_vertices, updated_edges)
  
  # Initiate vertices_with_agg_essages
  emptyRDD = spark.sparkContext.emptyRDD()
  from pyspark.sql.types import StructType, StructField, StringType, ArrayType, StringType, LongType
  schema = StructType([
    StructField('id', StringType(), True),
    StructField('aggregrated_parents_from_parent', ArrayType(StringType(), True)),
    StructField('iteration', LongType(), True)
  ])
  vertices_with_agg_essages = spark.createDataFrame(emptyRDD,schema)
  
  
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


    vertices_with_agg_essages = vertices_with_agg_essages.union(agg)
    
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
  helper = (vertices_with_agg_essages.
    groupBy("id").
    agg(f.max("iteration").alias("iteration")))

  vertices_with_agg_essages = helper.join(vertices_with_agg_essages, ["id", "iteration"],  how="inner")

  # Subset to furthermost children
  in_degress_df = g.inDegrees
  raw_df = (vertices.
   join( in_degress_df, ["id"], how='left_anti'))
  vertices_with_agg_essages = (raw_df.
                               join(vertices_with_agg_essages, ["id"],how="inner").select(f.col("id"),f.col("aggregrated_parents_from_parent"))
                              )
  vertices_with_agg_essages = (vertices_with_agg_essages.
                                 withColumn("SKU", f.col("aggregrated_parents_from_parent").getItem(0)).
                                 select(f.col("id"), f.col("SKU"))
                              )
    
    
  return(vertices_with_agg_essages)

# COMMAND ----------

res = get_sku_for_raw(g)

# COMMAND ----------

display(res)

# COMMAND ----------

# MAGIC %md
# MAGIC # Develop an algorithm to find the raw material with the quantities they are in their SKU

# COMMAND ----------

# MAGIC %md
# MAGIC Develop the Algorith step by step

# COMMAND ----------

iteration = 1

# COMMAND ----------

# initiate vertices_with_agg_essages
emptyRDD = spark.sparkContext.emptyRDD()
from pyspark.sql.types import StructType, StructField, StringType, LongType
schema = StructType([
  StructField('id', StringType(), True),
  StructField('qty', LongType(), True),
  StructField('iteration', LongType(), True)
  ])
vertices_with_agg_essages = spark.createDataFrame(emptyRDD,schema)

# COMMAND ----------

vertices_with_agg_essages.show()

# COMMAND ----------

g_for_loop = g

# COMMAND ----------

#Let's try the first value

msgToSrc = AM.edge["qty"]

agg = g_for_loop.aggregateMessages(
 f.first(AM.msg).alias("qty_from_parent"),
 sendToSrc=msgToSrc,
 sendToDst=None
)

agg = agg.withColumn("iteration", lit(iteration))

# COMMAND ----------

agg.show()

# COMMAND ----------

#Update agg
vertices_with_agg_essages = vertices_with_agg_essages.union(agg)


# COMMAND ----------

vertices_with_agg_essages.show()

# COMMAND ----------

#Now joiun this to the edges by dst
edges_old = g_for_loop.edges
edges_update = (edges_old.
                  join(agg, edges_old['dst'] == agg['id'], "inner").
                  withColumn("qty", F.col("qty")*F.col("qty_from_parent")).
                  select(f.col('src'),f.col('dst'),f.col('qty'))
               )

# COMMAND ----------

edges_update.show()

# COMMAND ----------

# Update the graph
g_for_loop = GraphFrame(vertices, edges_update)

# COMMAND ----------

# Do it again
iteration+=1

# COMMAND ----------

iteration

# COMMAND ----------

# Do it again
msgToSrc = AM.edge["qty"]

agg = g_for_loop.aggregateMessages(
 f.first(AM.msg).alias("qty_from_parent"),
 sendToSrc=msgToSrc,
 sendToDst=None
)

agg = agg.withColumn("iteration", lit(iteration))

#Update agg
vertices_with_agg_essages = vertices_with_agg_essages.union(agg)

display(vertices_with_agg_essages)

# COMMAND ----------

edges_update.count()

# COMMAND ----------

# Therefore: One needs to dod a do while loop until edges_update.count() == 0 and then return agg

# COMMAND ----------

# MAGIC %md
# MAGIC Loop the above 

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, LongType

# COMMAND ----------

def find_furthermost_children_with_prod_of_qty(gx):
  
  
  # Initialization:
  msgToSrc = AM.edge["qty"]
  
  # Initiate the graph to be updated by each loop iteration
  g_for_loop = gx
  
  # initiate vertices_with_agg_essages
  emptyRDD = spark.sparkContext.emptyRDD()
  schema = StructType([
  StructField('id', StringType(), True),
  StructField('qty', LongType(), True),
  StructField('iteration', LongType(), True)
  ])
  vertices_with_agg_essages = spark.createDataFrame(emptyRDD,schema)
  
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
    agg = agg.withColumn("iteration", lit(iteration))
    vertices_with_agg_essages = vertices_with_agg_essages.union(agg)
    
  
    #Update edges accordingly
    edges_old = g_for_loop.edges
    edges_update = (edges_old.
                    join(agg, edges_old['dst'] == agg['id'], "inner").
                    withColumn("qty", F.col("qty")*F.col("qty_from_parent")).
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
  helper = (vertices_with_agg_essages.
    groupBy("id").
    agg(f.max("iteration").alias("iteration"))
         )

  vertices_with_agg_essages = helper.join(vertices_with_agg_essages, ["id", "iteration"],  how="inner")

  # Subset to furthermost children
  in_degress_df = g.inDegrees
  raw_df = (vertices.
   join( in_degress_df, ["id"], how='left_anti' )
  )
  vertices_with_agg_essages = raw_df.join(vertices_with_agg_essages, ["id"], how="inner").select(f.col("id"),f.col("qty"))
    
  #Return
  return(vertices_with_agg_essages)

# COMMAND ----------

res = find_furthermost_children_with_prod_of_qty(g)

# COMMAND ----------

display(res)
