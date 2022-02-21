# Databricks notebook source
# MAGIC %md
# MAGIC # Hierarchical Time Series Generator
# MAGIC This notebook simulates hierarchical time series data

# COMMAND ----------

#################################################
# Python Packages
#################################################
import pandas as pd
import numpy as np
import datetime

from dateutil.relativedelta import relativedelta
from dateutil import rrule

import string
import random


from pyspark.sql.functions import pandas_udf, PandasUDFType, concat_ws
from pyspark.sql.types import StructType,StructField, StringType, DateType

# COMMAND ----------

#################################################
# Parameters
#################################################
n=10 # Number of SKU's per product
ts_length_in_years = 3 # Length of a time series in years
corona_breakpoint = datetime.date(year=2020,month=3,day=1) # date infortmation: At which date do Corona effects into play
percentage_decrease_corona_from = 20 # date information: define the decline after Corona comes into play
percentage_decrease_corona_to = 7 # date infortmation: define the decline after Corona comes into play
trend_factor_before_corona = 100 # date infortmation: trend before Corona

# COMMAND ----------

#################################################
# Create a Product Table
#################################################

data = [("Long Range Lidar",  "LRR"),
    ("Short Range Lidar", "SRR"),
    ("Camera", "CAM"),
    ("Long Range Radar", "LRR"),
    ("Short Range Radar", "SRR")
  ]

schema = StructType([ \
    StructField("Product",StringType(),True), \
    StructField("SKU_Prefix",StringType(),True)
  ])
 
product_identifier_lookup = spark.createDataFrame(data=data,schema=schema)

display(product_identifier_lookup)

# COMMAND ----------

#################################################
# Create a product hierarchy by simulating SKU's for each product
#################################################

# Define schema of output data-frame
schema = StructType(  [StructField("SKU_Postfix", StringType(), True)] + product_identifier_lookup.schema.fields)

#Help-function to generate a random string
def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

  
# Create a Pandas UDF to simulate unique SKU's, i.e. n random strings without repetition
@pandas_udf(schema, PandasUDFType.GROUPED_MAP)

def id_seuqence_generator(pdf):
    
  res = set()
  while True:
    res.add(id_generator())
    if len(res) >= n:
      break
  
  pdf_out = pd.DataFrame()
  
  pdf_out["SKU_Postfix"] = list(res)
  
  pdf_out["Product"] = pdf["Product"].iloc[0]
  pdf_out["SKU_Prefix"] = pdf["SKU_Prefix"].iloc[0]
  
  
  return pdf_out


# Apply the Pandas UDF and clean up
product_hierarchy = ( \
  product_identifier_lookup \
  .groupby("SKU_Prefix", "Product") \
  .apply(id_seuqence_generator) \
  .withColumn("SKU", concat_ws('_',"SKU_Prefix","SKU_Postfix")) \
  .select("Product","SKU")
      )

# Check that the number of rows is what is expected
assert product_hierarchy.count() == (n * product_identifier_lookup.count()), "Number of rows in final table contradicts with input parameters"

display(product_hierarchy)

# COMMAND ----------

#################################################
# Create a Pandas Data Frame for common date information for each time series
#################################################
# End Date: Make it a Monday
end_date = datetime.datetime(2021, 7, 19)
end_date = end_date + datetime.timedelta(-end_date.weekday()) #Make sure to get the monday before

# Start date: Is a monday, since we will go back integer number of weeks
start_date = end_date + relativedelta(weeks= (-ts_length_in_years * 52))

# Make a sequence 
date_range = list(rrule.rrule(rrule.WEEKLY, dtstart=start_date, until=end_date))
date_range = [x.date() for x in date_range]

date_range = pd.DataFrame(date_range, columns =['Date'])


# Derive Corona Time Range, 0 for for pre-Corona and a count after corona
min_date = np.min(date_range.Date[ date_range.Date >=  corona_breakpoint ])
breakpoint = date_range.Date.searchsorted(min_date, side='left')
help_list = [0] * (breakpoint -1) + list(range(0, len(date_range) - breakpoint + 1))
assert len(help_list) == len(date_range), "Longth of help_list is ambigious"
date_range = date_range.assign(Corona_Breakpoint_Helper = help_list)

# Derive Corona correction factor for demand
percentage_decrease = [  percentage_decrease_corona_from -  ( (percentage_decrease_corona_from - percentage_decrease_corona_to ) / max(help_list) ) * k  if k > 0 else 0 for k in  help_list]
factor = [ 1 if k == 0 else (100 - k) / 100 for k in percentage_decrease]
date_range = date_range.assign(Corona_Factor=factor)

# Derive X-mas Correction factor for demand
date_range = date_range.assign(Week = pd.DatetimeIndex(date_range['Date']).isocalendar().week.tolist())

conditions_xmas = [
      date_range.Week == 51,
      date_range.Week >= 52,
      date_range.Week == 1,
      date_range.Week == 2,
      date_range.Week == 3,
      date_range.Week == 4
    ]

choices_xmas = [
  0.85,
  0.8,
  1.1,
  1.15,
  1.1,
  1.05
]

date_range[ "Factor_XMas" ] = np.select(conditions_xmas, choices_xmas, default= 1.0)


display(date_range)


# COMMAND ----------

#################################################
# Enhance te product table with parameters for simulating time series
#################################################

# Get a list of all products from the hierarchy table and generate a list 
from  pyspark.sql.types import FloatType, ArrayType, IntegerType
from pyspark.sql.functions import monotonically_increasing_id


# Define schema for new columns
schema = StructType([StructField("Variance_RN", FloatType(), True),
          StructField("Offset_RN", FloatType(), True),
          StructField("AR_Pars_RN", ArrayType(FloatType()), True),
          StructField("MA_Pars_RN", ArrayType(FloatType()), True)])


# Generate random numbers for the ARMA process
n_ = product_identifier_lookup.count()
variance_random_number = list(abs(np.random.normal(100, 50, n_)))
offset_random_number = list(np.maximum(abs(np.random.normal(10000, 5000, n_)), 4000))
ar_length_random_number = np.random.choice(list(range(1,4)), n_)
ar_parameters_random_number = [np.random.uniform(low=0.1, high=0.9, size=x) for x in ar_length_random_number] 
ma_length_random_number = np.random.choice(list(range(1,4)), n_)
ma_parameters_random_number = [np.random.uniform(low=0.1, high=0.9, size=x) for x in ma_length_random_number] 


# Collect in a dataframe
pdf_helper = (pd.DataFrame(variance_random_number, columns =['Variance_RN']). 
              assign(Offset_RN = offset_random_number).
              assign(AR_Pars_RN = ar_parameters_random_number).
              assign(MA_Pars_RN = ma_parameters_random_number) 
             )

# Append column-wise
spark_df_helper = spark.createDataFrame(pdf_helper, schema = schema)
spark_df_helper = spark_df_helper.withColumn("row_id", monotonically_increasing_id())
product_identifier_lookup = product_identifier_lookup.withColumn("row_id", monotonically_increasing_id())
product_identifier_lookup_extended = product_identifier_lookup.join(spark_df_helper, ("row_id")).drop("row_id")
product_identifier_lookup = product_identifier_lookup.drop("row_id")
product_hierarchy_extended = product_hierarchy.join(product_identifier_lookup_extended.drop("SKU_Prefix"), ["Product"], how = "inner")
assert product_identifier_lookup_extended.count() == product_identifier_lookup.count(), "Ambigious number of rows after join"

display(product_hierarchy_extended)

# COMMAND ----------

import statsmodels.api as sm
import matplotlib.pyplot as plt

from pyspark.sql.functions import col,when

from pyspark.sql.functions import row_number, sqrt, round

#################################################
# Generate an individual time series for each Product-SKU combination
#################################################

# function to generate an ARMA process
def generate_arma(arparams, maparams, var, offset, number_of_points, plot):
  ar = np.r_[1, arparams] 
  ma = np.r_[1, maparams] 
  y = sm.tsa.arma_generate_sample(ar, ma, number_of_points, scale=var, burnin= 3000) + offset
  #y = np.round(y).astype(int)
  


  if plot:
    x = np.arange(1, len(y) +1)
    plt.plot(x, y, color ="red")
    plt.show()
    
  return(y)


#Schema for output dataframe
schema = StructType(  product_hierarchy.schema.fields + 
                    [
                      StructField("Date", DateType(), True),
                      StructField("Demand", FloatType(), True),
                      StructField("Corona_Factor", FloatType(), True),
                      StructField("Factor_XMas", FloatType(), True),
                      StructField("Corona_Breakpoint_Helper", FloatType(), True),
                      StructField("Row_Number", FloatType(), True)
                      
                      
                    ])



# Generate an ARMA
#pdf = product_hierarchy_extended.toPandas().head(1)


# Generate a time series with random parameters
@pandas_udf(schema, PandasUDFType.GROUPED_MAP)

def time_series_generator_pandas_udf(pdf):
  out_df = date_range.assign(
   Demand = generate_arma(arparams = pdf.AR_Pars_RN.iloc[0], 
                          maparams= pdf.MA_Pars_RN.iloc[0], 
                          var = pdf.Variance_RN.iloc[0], 
                          offset = pdf.Offset_RN.iloc[0], 
                          number_of_points = date_range.shape[0], 
                          plot = False),
    Product = pdf.Product.iloc[0],
    SKU = pdf.SKU.iloc[0]
  )

  out_df = out_df[["Product", "SKU", "Date", "Demand", "Corona_Factor", "Factor_XMas", "Corona_Breakpoint_Helper"]]
  
  out_df["Row_Number"] = range(0,len(out_df))

  return(out_df)

# Apply the Pandas UDF and clean up
demand_df = ( 
  product_hierarchy_extended 
  .groupby("Product", "SKU") 
  .apply(time_series_generator_pandas_udf) 
  .withColumn("Demand" , col("Demand") * col("Corona_Factor")) 
  .withColumn("Demand", when(col("Corona_Breakpoint_Helper") == 0,   
                             col("Demand") + trend_factor_before_corona * sqrt(col("Row_Number"))) 
                        .otherwise( col("Demand")))  
  .withColumn("Demand" , col("Demand") * col("Factor_XMas"))
  .withColumn("Demand" , round(col("Demand")))
  .select(col("Product"), col("SKU"), col("Date"), col("Demand") )
   )


display(demand_df)

# COMMAND ----------

# Plot indivudual series
res_table = demand_df.toPandas()
all_combis = res_table[[ "Product" , "SKU" ]].drop_duplicates()
random_series_to_plot = pd.merge(  res_table,   all_combis.iloc[[random.choice(list(range(len(all_combis))))]] ,  on =  [ "Product" , "SKU" ], how = "inner" )
selected_product = random_series_to_plot[ 'Product' ].iloc[0]
selected_sku = random_series_to_plot[ 'SKU' ].iloc[0]
random_series_to_plot = random_series_to_plot[["Date","Demand"]]

#Plot
plt.plot_date(random_series_to_plot.Date, random_series_to_plot.Demand, linestyle='solid')
plt.gcf().autofmt_xdate()
plt.title(f"Product: {selected_product}, SKU: {selected_sku}.")
plt.xlabel('Date')
plt.ylabel('Demand')
plt.show()

# COMMAND ----------

# Plot a sepecific time series
display(demand_df.join(demand_df.sample(False, 1 / demand_df.count(), seed=0).limit(1).select("SKU"), on=["SKU"], how="inner"))

# COMMAND ----------

# MAGIC %md
# MAGIC # Register table in database

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE DATABASE IF NOT EXISTS demand_db;
# MAGIC DROP TABLE IF EXISTS demand_db.demand_df_delta;
# MAGIC CREATE TABLE demand_db.demand_df_delta USING DELTA LOCATION '/FileStore/tables/demand_forecasting_solution_accellerator/demand_df_delta/'

# COMMAND ----------

# Write the data 
demand_df.write \
.mode("overwrite") \
.format("delta") \
.save('/FileStore/tables/demand_forecasting_solution_accellerator/demand_df_delta/')


# COMMAND ----------


