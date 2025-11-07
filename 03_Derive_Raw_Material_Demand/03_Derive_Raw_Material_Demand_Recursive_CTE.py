# Databricks notebook source
# MAGIC %md
# MAGIC This notebook should run on a serverless DBSQL cluster. Create the widgets first on an interactive cluster and then switch to a serverless DBSQL cluster.

# COMMAND ----------

# MAGIC %md
# MAGIC # Map the forecasted demand to raw materials
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC Once the demand is forecasted, manufacturers need to purchase raw material and initiate production planning. This notebook shows how to translate future SKU demand into raw materials planning. More precisely, we will do a Bill of Material (BoM) resolution to map the forecasted demand for each SKU to the appropriate demand of raw materials that are needed to produce the finished good that is mapped to the SKU.

# COMMAND ----------

# MAGIC %md
# MAGIC *Prerequisite: Make sure to run 01_Introduction_And_Setup and 02_Fine_Grained_Demand_Forecasting_AI_Forecast before running this notebook.*
# MAGIC
# MAGIC While the previous notebook *(02_Fine_Grained_Demand_Forecasting_AI_Forecast)* demonstrated the benefits of Databricks' approach for forecasting with great speed and cost-effectiveness, in this part we show how to use Databricks' CTE functionality to traverse the manufacturing value chain backwards to find out how much raw material is needed for production.
# MAGIC
# MAGIC Key highlights for this notebook:
# MAGIC - CTE's

# COMMAND ----------

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
# MAGIC <img src="https://raw.githubusercontent.com/maxkoehlerdatabricks/demand_forcasting_sa/max/Pictures/operations_process_forwards.png" width="1500"/>

# COMMAND ----------

# MAGIC %md
# MAGIC # How does the Bill of Material Table in this solution accelerator look like?
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC In our example the Bill of Material table consists of all raw materials that a final or intermediate material number consists of, along with related quantities. 
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Create the widgets first on an interactive cluster and then switch to a DBSQL cluster
# MAGIC --CREATE WIDGET TEXT catalogName DEFAULT 'maxkoehler_demos';
# MAGIC --CREATE WIDGET TEXT dbName DEFAULT 'demand_db';

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS intermediate_bom_traversal;

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS intermediate_bom_traversal_2;

# COMMAND ----------

# MAGIC %sql
# MAGIC USE CATALOG IDENTIFIER(:catalogName);
# MAGIC
# MAGIC DECLARE OR REPLACE VARIABLE dbname_individual STRING;
# MAGIC
# MAGIC SET VAR (dbname_individual) = (
# MAGIC   SELECT 
# MAGIC   concat(:dbName, "_", replace(split_part(current_user(), '@', 1), '.', '')) AS dbname_individual
# MAGIC );
# MAGIC
# MAGIC USE SCHEMA IDENTIFIER(dbname_individual);

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM bom

# COMMAND ----------

# MAGIC %md
# MAGIC ## We explain the algorithm first based on a simple example

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TEMP VIEW edges AS
# MAGIC SELECT * FROM VALUES
# MAGIC   ('Raw1', 'Intermediate1', 5),
# MAGIC   ('Intermediate1', 'Intermediate2', 3),
# MAGIC   ('Intermediate2', 'FinishedProduct', 1),
# MAGIC   ('Raw2', 'Intermediate3', 5),
# MAGIC   ('Intermediate3', 'FinishedProduct', 1),
# MAGIC   ('FinishedProduct', 'SKU', 1)
# MAGIC AS edges(src, dst, qty)

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

# MAGIC %sql
# MAGIC SELECT DISTINCT dst as component, qty as total_qty, dst as sku FROM edges e LEFT ANTI JOIN (SELECT DISTINCT src FROM edges) AS e2 ON e.dst = e2.src

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT DISTINCT src FROM edges e LEFT ANTI JOIN (SELECT DISTINCT dst FROM edges) AS e2 ON e.src = e2.dst

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from edges;

# COMMAND ----------

# MAGIC %sql
# MAGIC WITH RECURSIVE bom_traversal AS (
# MAGIC     SELECT DISTINCT dst as component, qty as total_qty, dst as sku 
# MAGIC     FROM edges e 
# MAGIC     LEFT ANTI JOIN (SELECT DISTINCT src FROM edges) AS e2 
# MAGIC     ON e.dst = e2.src
# MAGIC     UNION ALL
# MAGIC     SELECT e.src as component, e.qty * b.total_qty, sku
# MAGIC     FROM bom_traversal AS b
# MAGIC     JOIN edges AS e
# MAGIC     ON b.component = e.dst 
# MAGIC )
# MAGIC SELECT * 
# MAGIC FROM bom_traversal 
# MAGIC WHERE component IN (
# MAGIC     SELECT DISTINCT src 
# MAGIC     FROM edges e 
# MAGIC     LEFT ANTI JOIN (SELECT DISTINCT dst FROM edges) AS e2 
# MAGIC     ON e.src = e2.dst
# MAGIC );

# COMMAND ----------

# MAGIC %md
# MAGIC ## We now apply the concept on the forecasted demand data set

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT Product, SKU, Date, Demand FROM part_level_demand_with_forecasts;

# COMMAND ----------

# MAGIC %md
# MAGIC The BoM does not contain the mapping to SKU's. Therefore, we add an artifical assembly step with quantity 1

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM bom;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM sku_mapper;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT DISTINCT SKU FROM part_level_demand_with_forecasts

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TEMP VIEW edges AS
# MAGIC SELECT 
# MAGIC     final_mat_number AS src, 
# MAGIC     sku AS dst, 
# MAGIC     1 AS qty 
# MAGIC   FROM sku_mapper
# MAGIC UNION
# MAGIC SELECT 
# MAGIC     material_in AS src, 
# MAGIC     material_out AS dst,
# MAGIC     qty
# MAGIC   FROM bom

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE intermediate_bom_traversal
# MAGIC USING DELTA
# MAGIC AS
# MAGIC WITH RECURSIVE bom_traversal AS (
# MAGIC     SELECT DISTINCT dst as component, qty as total_qty, dst as sku 
# MAGIC     FROM edges e 
# MAGIC     LEFT ANTI JOIN (SELECT DISTINCT src FROM edges) AS e2 
# MAGIC     ON e.dst = e2.src
# MAGIC     UNION ALL
# MAGIC     SELECT e.src as component, e.qty * b.total_qty, sku
# MAGIC     FROM bom_traversal AS b
# MAGIC     JOIN edges AS e
# MAGIC     ON b.component = e.dst 
# MAGIC )
# MAGIC SELECT * 
# MAGIC FROM bom_traversal 
# MAGIC WHERE component IN (
# MAGIC     SELECT DISTINCT src 
# MAGIC     FROM edges e 
# MAGIC     LEFT ANTI JOIN (SELECT DISTINCT dst FROM edges) AS e2 
# MAGIC     ON e.src = e2.dst
# MAGIC );

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE intermediate_bom_traversal_2
# MAGIC SELECT sku as SKU, component as RAW, total_qty as QTY_RAW FROM intermediate_bom_traversal;

# COMMAND ----------

# MAGIC %md
# MAGIC ### Derive the demand for raw material and save to Delta

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE forecast_raw AS
# MAGIC SELECT 
# MAGIC     d.Product, 
# MAGIC     d.SKU, 
# MAGIC     r.RAW,  
# MAGIC     d.Date, 
# MAGIC     d.Demand as Demand_SKU, 
# MAGIC     r.QTY_RAW, 
# MAGIC     d.Demand * r.QTY_RAW as Demand_RAW 
# MAGIC FROM 
# MAGIC     part_level_demand_with_forecasts d
# MAGIC INNER JOIN 
# MAGIC     intermediate_bom_traversal_2 r 
# MAGIC ON 
# MAGIC     d.SKU = r.SKU
# MAGIC ORDER BY 
# MAGIC     d.SKU, r.RAW, d.Date

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM forecast_raw

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS intermediate_bom_traversal;

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS intermediate_bom_traversal_2;
