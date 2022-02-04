# Databricks notebook source
# MAGIC %md
# MAGIC # Demand forecasting 
# MAGIC In manufacturing, demand forecasting has been very successful in the last few years before the pandemic. The demand series for products was a relatively low volatile process and the likelihood of material shortages, leading to a gap in the demand and the production output, was relatively small. Therefore, manufacturing companies interpreted the amount of shipped products as the “real” demand and used highly sophisticated statistical models to extrapolate it to the future. This resulted in 
# MAGIC 
# MAGIC - Improved sales planning
# MAGIC - Highly optimized safety stock that allowed maximizing turn-rates under the constraint of holding a fairly good service-delivery-performance
# MAGIC - An optimized production planning by tracing back production outputs to raw material level using the bill of materials
# MAGIC 
# MAGIC After the pandemic, many manufacturers experienced a typical yoyo effect. The demand suddenly dropped in march 2020 and recovered so fast, that manufacturers under-planned dramatically. The resulting increase in orders to lower-tier manufacturers has evoked the first phase of a supplier crisis. As a result, production output did not match the actual demand any more. The yoyo effect increased volatility in the series leading to an unjustified recommendation to increase safety stock. Production- and sales-planning was rather forced by the availability of raw materials than driven by the actual demand. Statistical demand planning was put to the test. 
# MAGIC 
# MAGIC A perfect example of this is the chip crisis. After first reducing and then increasing orders, car-manufacturers and -suppliers had to compete with the increased demand for semiconductors due to remote work. If that was not enough, a couple of events were to follow. The trade war of China and the United States imposed restrictions on China’s largest chip manufacturer. The Texas ice storm forced the closure of two plants. Taiwan experienced a severe drought which further reduced the supply. Two Japanese plants caught fire, one as a result of an earthquake.
# MAGIC Can statistical demand forecasting forecast the afore-mentioned force-majeure-events? Certainly not! However, we think that Databricks offers a very good platform to adapt to the new challenges.
# MAGIC 
# MAGIC - Collaborative notebooks can be used to visualize demand series and help when incorporating business knowledge fitting the time series.
# MAGIC - The power of parallelization helps scaling the knowledge to the series of thousands of products.
# MAGIC - Tracking experiments using MLFlow leads to an improved re-usability.
# MAGIC - The incorporation of features, accessible by all personas in the company, help fitting the time series in a better way.
# MAGIC 
# MAGIC In this solution accelerator, we will show-case the benefits of using Databricks on a simulated data set. We assume a tier one car manufacturer that produces advanced driver assistance systems. We use collaborative notebooks to explore a set of time series models. We then scale single-series-models using Spark’s power of parallelization.
