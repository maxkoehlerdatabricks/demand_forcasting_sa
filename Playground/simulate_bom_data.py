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
# MAGIC # Goal:
# MAGIC - For each Prodcuct we need a pattern of a bom as an edge data frame
# MAGIC - The data frame needs to have a quantity
# MAGIC - Certain edges woould be repeadet
# MAGIC - The nodes must be in the same material number format
# MAGIC - Try in Python first

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



# COMMAND ----------

A = g.get_edgelist()
g_nx = nx.DiGraph(A) # In case your graph is directed

# COMMAND ----------

#nx.to_pandas_dataframe(g_nx)
nx.to_pandas_edgelist(g_nx)

# COMMAND ----------



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

import string

# COMMAND ----------

def generate_random_strings(n):
  n = 1000
  random.seed(123)
  random_mat_numbers = set()
  while True:
    random_mat_numbers.add(id_generator(size=5))
    if len(random_mat_numbers) >= n:
      break
  return(random_mat_numbers)

# COMMAND ----------

generate_random_strings(1000)

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



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------


