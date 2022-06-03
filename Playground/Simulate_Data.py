# Databricks notebook source
# MAGIC %md
# MAGIC This is from https://www.kaggle.com/c/demand-forecasting-kernels-only/data?select=train.csv \
# MAGIC Goal: Try to apply https://scikit-hts.readthedocs.io/en/latest/ to the datasets

# COMMAND ----------

dbutils.fs.ls ("/FileStore/tables/demand_forecasting_solution_accellerator/hierarchical_forecasting/")

# COMMAND ----------

# MAGIC %md 
# MAGIC Uploaded the data to here <br>
# MAGIC /FileStore/tables/demand_forecasting_solution_accellerator/hierarchical_forecasting/sample_submission.csv <br>
# MAGIC /FileStore/tables/demand_forecasting_solution_accellerator/hierarchical_forecasting/test.csv <br>
# MAGIC /FileStore/tables/demand_forecasting_solution_accellerator/hierarchical_forecasting/train.csv 

# COMMAND ----------

# MAGIC %md Documentation for hts paython package
# MAGIC https://scikit-hts.readthedocs.io/en/latest/

# COMMAND ----------

# Packages 
import pandas as pd

# COMMAND ----------

#Read in data
dat_train = pd.read_csv("/dbfs/FileStore/tables/demand_forecasting_solution_accellerator/hierarchical_forecasting/train.csv")

display(dat_train)

# COMMAND ----------

# MAGIC %md
# MAGIC The goal is to find a hierarchy adn modify the data set so that a standard hts-way of analysing using is possible. Once this is done visualise and talk to Bala

# COMMAND ----------

# MAGIC %md
# MAGIC Hierarchy
# MAGIC We need an SKU (not a GTIN). A Product Identifier. This a two level hierarchy.
# MAGIC 
# MAGIC Product Identifier:
# MAGIC long range lidar
# MAGIC short range lidar
# MAGIC camera
# MAGIC short range radar
# MAGIC long range radar
# MAGIC 
# MAGIC 
# MAGIC SKU - 1000 for each product identifier:
# MAGIC LRL - Random 6 digit string
# MAGIC SRL
# MAGIC CAM
# MAGIC SSR
# MAGIC LRR
# MAGIC 
# MAGIC Each SKU has a time series that is highly fluctuating and has a different altitide
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC To Do:
# MAGIC 
# MAGIC - Automatically assign Identifiers
# MAGIC - Time range 3 years
# MAGIC - Simulate a time series for each SKU automatically

# COMMAND ----------

# MAGIC %md
# MAGIC # Generating Hierarchy Metadata

# COMMAND ----------

import pandas as pd

product_identifiers = ["long range lidar", "short range lidar", "camera", "short range radar", "long range radar"]
prefix = ["LRR", "SRR", "CAM", "SRR", "LRR"]

product_identifiers_lookup = pd.DataFrame(list(zip(product_identifiers, prefix)),
               columns =['Product', 'SKU_prefix'])

product_identifiers_lookup

# COMMAND ----------

import string
import random

#Generate a random string
def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))
  
# Generate a list of n random strings without repetition
def id_seuqence_generator(n):
  res = set()
  while True:
    res.add(id_generator())
    if len(res) >= n:
      break
  return list(res)

#id_seuqence_generator(n=1000)

def product_sku_generator(sku_per_product):

  res = []

  for product in product_identifiers:
    df = pd.DataFrame(id_seuqence_generator(sku_per_product))
    df.columns = ['SKU_postfix']
    df['Product'] = product

    df = pd.merge(df, product_identifiers_lookup, how='inner')

    df['SKU'] = df['SKU_prefix'].str.cat(df['SKU_postfix'])

    df = df.drop(columns = ['SKU_postfix','SKU_prefix'])

    res.append(df)

  df_all = pd.concat(res, ignore_index=True)
  
  return df_all

# product_sku_generator(sku_per_product=10)


# COMMAND ----------

product_sku_generator(sku_per_product=10)

# COMMAND ----------

# MAGIC %md
# MAGIC # Generate demand values of time series data

# COMMAND ----------

# Packages
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd

# COMMAND ----------

#arparams = np.array([.001, -.001])
#maparams = np.array([.001, .001])
#var = 100
#offset = 10000
#number_of_points = 250

def generate_arma(arparams, maparams, var, offset, number_of_points, plot):
  ar = np.r_[1, arparams] 
  ma = np.r_[1, maparams] 
  y = sm.tsa.arma_generate_sample(ar, ma, number_of_points, scale=var, burnin= 3000) + offset
  y = np.round(y).astype(int)
  


  if plot:
    x = np.arange(1, len(y) +1)
    plt.plot(x, y, color ="red")
    plt.show()
    
  return(y)

# COMMAND ----------

generate_arma(arparams=np.array([.7, -.2]), maparams= np.array([.7, .2]), var = 100, offset = 10000, number_of_points = 100, plot = True)

# COMMAND ----------

# MAGIC %md
# MAGIC <span style="color:red">What to do next: Add a seasonality, can be just triggered by dummies for certain events, can be because of promotions???</span>

# COMMAND ----------

# Generate a time series with random parameters
number_of_points_fixed = 100

# for every Product / SKU combination:
variance_random_number = abs(np.random.normal(100, 50, 1)[0])
offset_random_number = max(abs(np.random.normal(10000, 5000, 1)[0]), 4000)
ar_length_random_number = random.choice(list(range(1,4)))
ar_parameters_random_number = np.random.uniform(low=0.1, high=0.9, size=ar_length_random_number)
ma_length_random_number = random.choice(list(range(1,4)))
ma_parameters_random_number = np.random.uniform(low=0.1, high=0.9, size=ma_length_random_number)

generate_arma(arparams=ar_parameters_random_number, maparams= ma_parameters_random_number, var = variance_random_number, offset = offset_random_number, number_of_points = number_of_points_fixed, plot = False)

# COMMAND ----------

# MAGIC %md
# MAGIC # Generate time range of time series data

# COMMAND ----------

from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
from dateutil import rrule

def date_sequence_of_mondays(end_date_go_back_months, start_date_go_back_years_from_end_date):

  # End Date: Make it a Monday
  end_date = date.today() + relativedelta(months=-end_date_go_back_months)
  end_date = end_date + timedelta(-end_date.weekday())

  # Start date: Is a monday, since we will go back integer number of weeks
  start_date = end_date + relativedelta(weeks= (-start_date_go_back_years_from_end_date * 52))

  # Make a sequence 
  data_seq_lst = list(rrule.rrule(rrule.WEEKLY, dtstart=start_date, until=end_date))
  data_seq_lst = [x.date() for x in data_seq_lst]
  data_seq_lst = np.asarray(data_seq_lst)
  return(data_seq_lst)

date_range = date_sequence_of_mondays(end_date_go_back_months=6, start_date_go_back_years_from_end_date=3)

date_range

# COMMAND ----------

# MAGIC %md
# MAGIC # Combine hierarchy, time series time range and time series values by generating a time series with date and demand column for each SKU product combination from the hierarchy

# COMMAND ----------

skus_per_product = 100

# Generate a table
hierarchy_table = product_sku_generator(sku_per_product = skus_per_product)

# Get a list of all products from the hierarchy table and generate a list 
n_ = product_identifiers_lookup.shape[0]

variance_random_number = list(abs(np.random.normal(100, 50, n_)))
offset_random_number = list(np.maximum(abs(np.random.normal(10000, 5000, n_)), 4000))
ar_length_random_number = np.random.choice(list(range(1,4)), n_)
ar_parameters_random_number = [np.random.uniform(low=0.1, high=0.9, size=x) for x in ar_length_random_number] 
ma_length_random_number = np.random.choice(list(range(1,4)), n_)
ma_parameters_random_number = [np.random.uniform(low=0.1, high=0.9, size=x) for x in ma_length_random_number] 


product_identifiers_lookup_appended = (product_identifiers_lookup.
 assign(variance_random_number = variance_random_number).
 assign(offset_random_number = offset_random_number).
 assign(ar_parameters_random_number = ar_parameters_random_number).
 assign(ma_parameters_random_number = ma_parameters_random_number)                                 
)

hierarchy_table_appended = pd.merge(hierarchy_table, product_identifiers_lookup_appended, how="left")

hierarchy_table_appended['series'] =  np.arange(len(hierarchy_table_appended))

res = [ ]

for i in np.arange(len(hierarchy_table_appended)):
  ts_array = generate_arma(arparams=hierarchy_table_appended['ar_parameters_random_number'].loc[i], 
                         maparams= hierarchy_table_appended['ma_parameters_random_number'].loc[i], 
                         var = hierarchy_table_appended['variance_random_number'].loc[i], 
                         offset = hierarchy_table_appended['offset_random_number'].loc[i], 
                         number_of_points = len(date_range), 
                         plot = False)


  single_time_series = pd.DataFrame(date_range, columns = ['date']).assign(demand = ts_array, series = i)

  res.append(single_time_series)
  
ts_table = pd.concat(res)

hierarchy_ts_table = pd.merge(hierarchy_table_appended, ts_table, how='inner', on = 'series')
hierarchy_ts_table = hierarchy_ts_table[list(hierarchy_table.columns)+ ['date', 'demand']]

assert (len(product_identifiers_lookup) * skus_per_product * len(date_range) == len(hierarchy_ts_table)), "Number of rows in final table does not reflect input parameters"

# COMMAND ----------

# MAGIC %md
# MAGIC # Model Christmas and Corona shocks and trends for individual series

# COMMAND ----------

#Select random series
all_combis = hierarchy_ts_table[[ "Product" , "SKU" ]].drop_duplicates()
random_series_to_plot = pd.merge(  hierarchy_ts_table,   all_combis.iloc[[random.choice(list(range(len(all_combis))))]] ,  on =  [ "Product" , "SKU" ], how = "inner" )
selected_product = random_series_to_plot[ 'Product' ].iloc[0]
selected_sku = random_series_to_plot[ 'SKU' ].iloc[0]
random_series_to_plot = random_series_to_plot[["date","demand"]]

#Plot
plt.plot_date(random_series_to_plot.date, random_series_to_plot.demand, linestyle='solid')
plt.gcf().autofmt_xdate()
plt.title(f"Product: {selected_product}, SKU: {selected_sku}.")
plt.xlabel('Date')
plt.ylabel('Demand')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Corona shock
# MAGIC From 1.3.2020 demand drops by - 30 % 
# MAGIC Then revovery to a level by - 15 %
# MAGIC Say that revovery rate is linear, i.e. 
# MAGIC n = number of points between 1.3.2020 and last point, enumerate 1 to n
# MAGIC percentage_dercrease = ((30 - 15) / n ) * k for the k'th point
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC Christmas shock
# MAGIC Every second moday of december - 10 %
# MAGIC Every third and fourth monday - 80 %
# MAGIC Then normal
# MAGIC 
# MAGIC Increasing trend before these shocks
# MAGIC y = y + 30 * np.sqrt(np.asarray(range(0,len(y))))

# COMMAND ----------

import datetime

##################################################
# Parameters
##################################################
percentage_decrease_corona_from = 20
percentage_decrease_corona_to = 7

trend_factor_before_corona = 100




##################################################
# Select random series
##################################################
all_combis = hierarchy_ts_table[[ "Product" , "SKU" ]].drop_duplicates()
random_series_to_plot = pd.merge(  hierarchy_ts_table,   all_combis.iloc[[random.choice(list(range(len(all_combis))))]] ,  on =  [ "Product" , "SKU" ], how = "inner" )
selected_product = random_series_to_plot[ 'Product' ].iloc[0]
selected_sku = random_series_to_plot[ 'SKU' ].iloc[0]
random_series_to_plot = random_series_to_plot[["date","demand"]]


##################################################
# Derive Corona Time Range
##################################################
min_date = np.min(random_series_to_plot.date[ random_series_to_plot.date >=  datetime.date(year=2020,month=3,day=1) ])
breakpoint = random_series_to_plot.date.searchsorted(min_date, side='left')
help_list = [0] * (breakpoint -1) + list(range(0, len(random_series_to_plot) - breakpoint + 1))


##################################################
# Corona Effect
##################################################
assert len(help_list) == len(random_series_to_plot), "length not equal"
assert help_list[(breakpoint-1) : (breakpoint+2)], "breakpoint ambitious"
percentage_decrease = [  percentage_decrease_corona_from -  ( (percentage_decrease_corona_from - percentage_decrease_corona_to ) / max(help_list) ) * k  if k > 0 else 0 for k in  help_list]
factor = [ 1 if k == 0 else (100 - k) / 100 for k in percentage_decrease]

# apply the factor
random_series_to_plot = random_series_to_plot.assign(factor=factor)
random_series_to_plot['demand'] = random_series_to_plot['demand'] * random_series_to_plot['factor']

##################################################
# Add a trend before Corona
##################################################
choicelist = [
  random_series_to_plot['demand'] + trend_factor_before_corona * np.sqrt(np.asarray(range(0,len(random_series_to_plot)))),
  random_series_to_plot['demand']
]
condlist = [
  np.asarray(help_list) == 0,
  np.asarray(help_list) > 0
]
random_series_to_plot['demand'] = np.select(condlist, choicelist, default=0)


##################################################
# Add a Xmas Effect
##################################################


random_series_to_plot= random_series_to_plot.assign(week = pd.DatetimeIndex(random_series_to_plot['date']).isocalendar().week.tolist())

conditions_xmas = [
  random_series_to_plot.week == 51,
  random_series_to_plot.week >= 52,
  random_series_to_plot.week == 1,
  random_series_to_plot.week == 2,
  random_series_to_plot.week == 3,
  random_series_to_plot.week == 4
]

choices_xmas = [
  0.85,
  0.8,
  1.1,
  1.15,
  1.1,
  1.05
]


random_series_to_plot[ "factor_xmas" ] = np.select(conditions_xmas, choices_xmas, default= 1.0)
random_series_to_plot['demand'] = random_series_to_plot['factor_xmas'] * random_series_to_plot['demand']


##################################################
# Plot
##################################################
plt.plot_date(random_series_to_plot.date, random_series_to_plot.demand, linestyle='solid')
plt.gcf().autofmt_xdate()
plt.title(f"Product: {selected_product}, SKU: {selected_sku}.")
plt.xlabel('Date')
plt.ylabel('Demand')
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC # Model Christmas and Corona shocks and trends and iterate over individual series

# COMMAND ----------

##################################################
# Parameters
##################################################
percentage_decrease_corona_from = 20
percentage_decrease_corona_to = 7

trend_factor_before_corona = 100

all_combis = hierarchy_ts_table[[ "Product" , "SKU" ]].drop_duplicates()

res = [ ]

for index, row in all_combis.iterrows():
  
    #print(row['Product'])
    #print(row[ 'SKU' ])
    #print(type(row['Product']))
    #print(type(row[ 'SKU' ]))
    
    #print(hierarchy_ts_table[ 'Product' ]  ==  row['Product'])
    #print(hierarchy_ts_table[ 'SKU' ] == row['SKU'])
    
      
    ss = ((hierarchy_ts_table[ 'Product' ]  ==  row['Product']) & (hierarchy_ts_table[ 'SKU' ] == row['SKU']))
    random_series_to_plot = hierarchy_ts_table[ ss ]
    
    
    ##################################################
    # Derive Corona Time Range
    ##################################################
    min_date = np.min(random_series_to_plot.date[ random_series_to_plot.date >=  datetime.date(year=2020,month=3,day=1) ])
    breakpoint = random_series_to_plot.date.searchsorted(min_date, side='left')
    help_list = [0] * (breakpoint -1) + list(range(0, len(random_series_to_plot) - breakpoint + 1))


    ##################################################
    # Corona Effect
    ##################################################
    assert len(help_list) == len(random_series_to_plot), "length not equal"
    assert help_list[(breakpoint-1) : (breakpoint+2)], "breakpoint ambitious"
    percentage_decrease = [  percentage_decrease_corona_from -  ( (percentage_decrease_corona_from - percentage_decrease_corona_to ) / max(help_list) ) * k  if k > 0 else 0 for k in  help_list]
    factor = [ 1 if k == 0 else (100 - k) / 100 for k in percentage_decrease]

    # apply the factor
    random_series_to_plot = random_series_to_plot.assign(factor=factor)
    random_series_to_plot['demand'] = random_series_to_plot['demand'] * random_series_to_plot['factor']

    ##################################################
    # Add a trend before Corona
    ##################################################
    choicelist = [
      random_series_to_plot['demand'] + trend_factor_before_corona * np.sqrt(np.asarray(range(0,len(random_series_to_plot)))),
      random_series_to_plot['demand']
    ]
    condlist = [
      np.asarray(help_list) == 0,
      np.asarray(help_list) > 0
    ]
    random_series_to_plot['demand'] = np.select(condlist, choicelist, default=0)


    ##################################################
    # Add a Xmas Effect
    ##################################################
    
    random_series_to_plot= random_series_to_plot.assign(week = pd.DatetimeIndex(random_series_to_plot['date']).isocalendar().week.tolist())
    
    conditions_xmas = [
      random_series_to_plot.week == 51,
      random_series_to_plot.week >= 52,
      random_series_to_plot.week == 1,
      random_series_to_plot.week == 2,
      random_series_to_plot.week == 3,
      random_series_to_plot.week == 4
    ]

    choices_xmas = [
      0.85,
      0.8,
      1.1,
      1.15,
      1.1,
      1.05
    ]

    random_series_to_plot= random_series_to_plot.assign(week = pd.DatetimeIndex(random_series_to_plot['date']).isocalendar().week.tolist())
    random_series_to_plot[ "factor_xmas" ] = np.select(conditions_xmas, choices_xmas, default= 1.0)
    random_series_to_plot['demand'] = random_series_to_plot['factor_xmas'] * random_series_to_plot['demand']
    
    random_series_to_plot = random_series_to_plot[['Product' , 'SKU', 'date',	'demand']]
    
    res.append(random_series_to_plot)
    
    
res_table = pd.concat(res)
assert(len(res_table) == len(hierarchy_ts_table))


# COMMAND ----------

# Plot indivudual series
all_combis = res_table[[ "Product" , "SKU" ]].drop_duplicates()
random_series_to_plot = pd.merge(  res_table,   all_combis.iloc[[random.choice(list(range(len(all_combis))))]] ,  on =  [ "Product" , "SKU" ], how = "inner" )
selected_product = random_series_to_plot[ 'Product' ].iloc[0]
selected_sku = random_series_to_plot[ 'SKU' ].iloc[0]
random_series_to_plot = random_series_to_plot[["date","demand"]]

#Plot
plt.plot_date(random_series_to_plot.date, random_series_to_plot.demand, linestyle='solid')
plt.gcf().autofmt_xdate()
plt.title(f"Product: {selected_product}, SKU: {selected_sku}.")
plt.xlabel('Date')
plt.ylabel('Demand')
plt.show()

# COMMAND ----------

res_table

# COMMAND ----------

# MAGIC %md
# MAGIC # Convert to a Delta Table

# COMMAND ----------

from pyspark.sql.types import StructType,StructField, StringType, FloatType, DateType

mySchema = StructType([ StructField("Product", StringType(), True)\
                       ,StructField("SKU", StringType(), True) \
                       ,StructField("date", DateType(), True) \
                       ,StructField("demand", FloatType(), True)])

hierarchical_ts_df = spark.createDataFrame(res_table,schema=mySchema)
hierarchical_ts_df.printSchema()
display(hierarchical_ts_df)

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE DATABASE IF NOT EXISTS "demand_db";

# COMMAND ----------

# Define the input and output formats and paths and the table name.
write_format = 'delta'
save_path = '/FileStore/tables/demand_forecasting_solution_accellerator/hierarchical_forecasting2/'
table_name = 'hierarchical_ts_table'

# Write the data to its target.
hierarchical_ts_df.write \
.mode("overwrite") \
.format(write_format) \
.save(save_path)



# Create the table.
spark.sql("CREATE TABLE " +  "demand_db." + table_name + " USING DELTA LOCATION '" + save_path + "'")

# COMMAND ----------

# MAGIC %md
# MAGIC # Some Pandas UDF Tests

# COMMAND ----------

# the input would be a Spark dataframe to apply a group by on
data = [("James","M"),("Michael","M"),
        ("Maria","F")]

columns = ["name","gender"]
df = spark.createDataFrame(data = data, schema = columns)
df.show()

# COMMAND ----------

# However, into the function goes a Pandas Dataframe
pdf = df.toPandas()

# COMMAND ----------

# The columns of the output data frame, which is also a Pandas Data Frame must exactly match the specified schema, including the sequence of the columns names
d = {'value1': [1, 2], 'value2': [3, 4]}
df_infunction = pd.DataFrame(data=d)
df_infunction["name"] = pdf['name'].iloc[0]
df_infunction['gender'] = pdf['gender'].iloc[0]
df_infunction

# COMMAND ----------

# Note that the sequence and colum names of schema coincides with Pandas Data Frame output of function
from pyspark.sql.types import StructType, StructField, StringType, FloatType
target_schema = StructType([ \
    StructField("value1",FloatType(),True), \
    StructField("value2",FloatType(),True), \ 
    StructField("name",StringType(),True), \
    StructField("gender",StringType(),True) 
  ])

# COMMAND ----------

from pyspark.sql.functions import pandas_udf, PandasUDFType
import pandas as pd

@pandas_udf(target_schema, PandasUDFType.GROUPED_MAP)
# Input/output are both a pandas.DataFrame
def create_ts(pdf):
    d = {'value1': [1, 2], 'value2': [3, 4]}
    df_infunction = pd.DataFrame(data=d)
    df_infunction['name'] = pdf['name'].iloc[0]
    df_infunction['gender'] = pdf['gender'].iloc[0]
    
    return df_infunction 

tmp = df.groupby('name','gender').apply(create_ts)
tmp.show()
