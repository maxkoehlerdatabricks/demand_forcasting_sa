# Databricks notebook source
# MAGIC %md
# MAGIC # Part-Level Demand forecasting 
# MAGIC 
# MAGIC Demand forecasting forms one of the core applications for manufacturers. Manufacturers require a reasonable forecast in order to 
# MAGIC 1. plan the scaling of manufacturing operations
# MAGIC 2. ensure sufficient inventory
# MAGIC 3. guarantee customer fulfillment. 
# MAGIC 
# MAGIC Part-level demand forecasting is especially important in discrete manufacturing where manufacturers are at the mercy of their supply chain. In recent years, manufacturers have been investing heavily in quantitative-based forecasting that is driven by historical data and powered using either statistical or machine learning techniques. 
# MAGIC Demand forecasting has proven to be very successful in pre-pandemic years. The demand series for products had relatively low volatility and the likelihood of material shortages, leading to a gap in the demand and the production output, was relatively small. Therefore, manufacturers interpreted the number of shipped products as the “true” demand and used highly sophisticated statistical models to extrapolate it to the future. This resulted in 
# MAGIC 1. Improved sales planning
# MAGIC 2. Highly optimized safety stock that allowed maximizing turn-rates under the constraint of holding a fairly good service-delivery-performance
# MAGIC 3. An optimized production planning by tracing back production outputs to raw material level using the bill of materials (BoM)
# MAGIC 
# MAGIC However, since the pandemic, demand has seen huge volatility and fluctuations. Demand dropped hugely in the early days, led by a V-shaped recovery that resulted in underplanning. The resulting increase in orders to lower-tier manufacturers has evoked the first phase of a supplier crisis. As a result, production output did not match the actual demand any more. Fluctuations increased volatility in the series leading to an unjustified recommendation to increase safety stock. Production and sales planning was forced by the availability of raw materials rather than driven by the actual demand. Statistical demand planning was put to the test. 
# MAGIC A perfect example is the chip crisis. After first reducing and then increasing orders, car manufacturers and suppliers had to compete with the increased demand for semiconductors due to remote work. If that was not enough, a couple of events drove further volatility. The trade war between China and the United States imposed restrictions on China’s largest chip manufacturer. The Texas ice storm of 2021 resulted in a power crisis that forced the closure of several computer-chip facilities; Texas is the center of semiconductor manufacturing in the US. Taiwan experienced a severe drought which further reduced the supply. Two Japanese plants caught fire, one as a result of an earthquake.
# MAGIC 
# MAGIC Ref regarding Boom & Bust Cycles in Chips: https://www.economist.com/business/2022/01/29/when-will-the-semiconductor-cycle-peak 
# MAGIC 
# MAGIC Can statistical demand forecasting forecast the aforementioned force majeure events? Certainly not! However, we think that Databricks offers a very good platform to build large-scale forecasting solutions that will offer sufficient guidance to help manufacturers maneuver through these challenges. 
# MAGIC - Collaborative notebooks can be used to visualize demand series and help when incorporating business knowledge fitting the time series.
# MAGIC - The power of parallelization helps scale the knowledge to a series of thousands of products.
# MAGIC - Tracking experiments using MLFlow leads to an improved re-usability.
# MAGIC - The incorporation of features, accessible by all personas in the company, helps fit the time series in a better way.
# MAGIC 
# MAGIC In this solution accelerator, we will show-case the benefits of using Databricks on a simulated data set. We assume a tier one automotive manufacturer that produces advanced driver assistance systems. We use collaborative notebooks to explore a set of time series models. We then demonstrate the ability to train and deploy the models at scale using Spark’s ability to parallelize.
