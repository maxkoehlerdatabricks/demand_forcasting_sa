# Databricks notebook source
# MAGIC %md
# MAGIC This notebook is tested on a an ML enabled cluster, DBR 17.2

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Introduction
# MAGIC Demand forecasting is an integral business process for manufacturers. Manufacturers require accurate forecasts in order to:  
# MAGIC 1. plan the scaling of manufacturing operations
# MAGIC 2. ensure sufficient inventory
# MAGIC 3. guarantee customer fulfillment
# MAGIC
# MAGIC
# MAGIC Part-level demand forecasting is especially important in discrete manufacturing where manufacturers are at the mercy of their supply chain. In recent years, manufacturers have been investing heavily in quantitative-based forecasting that is driven by historical data and powered using either statistical or machine learning techniques. 
# MAGIC
# MAGIC
# MAGIC Demand forecasting has proven to be very successful in pre-pandemic years. The demand series for products had relatively low volatility and the likelihood of material shortages was relatively small. Therefore, manufacturers simply interpreted the number of shipped products as the “true” demand and used highly sophisticated statistical models to extrapolate into the future. This previously provided:  
# MAGIC - Improved sales planning
# MAGIC - Highly optimized safety stock that allowed maximizing turn-rates, provided fairly good service-delivery performance
# MAGIC - An optimized production planning by tracing back production outputs to raw material level using the bill of materials (BoM)
# MAGIC
# MAGIC
# MAGIC However, since the pandemic, demand has seen huge volatility and fluctuations. Demand dropped hugely in the early days, led by a V-shaped recovery that resulted in underplanning. The resulting increase in orders to lower-tier manufacturers in fact evoked the first phase of a supplier crisis. In essence, production output no longer matched actual demand, with any increases in volatility often leading to unjustified recommendations to increase safety stock. Production and sales planning were forced by the availability of raw materials rather than driven by the actual demand. Standard demand planning approaches were approaching major limits.
# MAGIC
# MAGIC
# MAGIC A perfect example can be found in the chip crisis. After first reducing and then increasing orders, car manufacturers and suppliers have had to compete with the increased demand for semiconductors due to remote work. To make matters worse, several significant events drove volatility even further. The trade war between China and the United States imposed restrictions on China’s largest chip manufacturer. The Texas ice storm of 2021 resulted in a power crisis that forced the closure of several computer-chip facilities; Texas is the center of semiconductor manufacturing in the US. Taiwan experienced a severe drought which further reduced the supply. Two Japanese plants caught fire, one as a result of an earthquake.
# MAGIC *Reference: Boom & Bust Cycles in Chips (https://www.economist.com/business/2022/01/29/when-will-the-semiconductor-cycle-peak)* 
# MAGIC
# MAGIC
# MAGIC **Could statistical demand forecasting have predicted the aforementioned ‘force majeure’ events? 
# MAGIC Certainly not! However, we think that Databricks offers an excellent platform to build large-scale forecasting solutions to help manufacturers maneuver through these challenges.**
# MAGIC
# MAGIC In this solution accelerator we demonstrate
# MAGIC - How to effectively scale demand forecasting for thousands of SKU's using ai_forecast()
# MAGIC - How to leverage SAP data to pull Bill of Material tables to then translate the forecasted SKU level demand to the desired raw material demand. In this section we will use Recursive Common Table Expressions (CTEs) for the BoM resolution
# MAGIC - How manage supplier shortages by optimizing the raw material inventory and providing transparency of what can delivered to the end customer. In this section we will use Delta Live Tables
# MAGIC - How to allow non technical staff from you purchasing, ptoduction or logistics consume by leveraging a conversaational experience instead of code. in this section we will use AI/BI Dashboards and Genie. 

# COMMAND ----------

# MAGIC %md
# MAGIC # Setup

# COMMAND ----------

dbutils.widgets.dropdown('reset_all_data', 'false', ['true', 'false'], 'Reset all data')
dbutils.widgets.text('catalogName',  'maxkoehler_demos' , 'Catalog Name')
dbutils.widgets.text('dbName',  'demand_db' , 'Database Name')

# COMMAND ----------

catalogName = dbutils.widgets.get('catalogName')
dbName = dbutils.widgets.get('dbName')
reset_all_data = dbutils.widgets.get('reset_all_data') == 'true'

# COMMAND ----------

# MAGIC %md
# MAGIC **⚠️ Needs to be run using ML Runtime due to use of MLFlow**

# COMMAND ----------

# MAGIC %run ./_resources/00-setup $reset_all_data=$reset_all_data $catalogName=$catalogName $dbName=$dbName

# COMMAND ----------

# MAGIC %md
# MAGIC # Understanding the data

# COMMAND ----------

demand_df = spark.read.table(f"{catalogName}.{dbName}.part_level_demand")

# COMMAND ----------

display(demand_df.select("Product").dropDuplicates())
