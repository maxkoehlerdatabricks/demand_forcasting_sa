{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "application/vnd.databricks.v1+cell": {
     "cellMetadata": {
      "byteLimit": 2048000,
      "rowLimit": 10000
     },
     "inputWidgets": {},
     "nuid": "def8ee1d-55fd-4e87-b70d-cc951874ddfb",
     "showTitle": false,
     "tableResultSettingsMap": {},
     "title": ""
    }
   },
   "source": [
<<<<<<< Updated upstream
    "This notebook is tested on a serverless DBSQL cluster"
=======
    "This notebook is tested on a Serverless DBSQL Cluster"
>>>>>>> Stashed changes
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "application/vnd.databricks.v1+cell": {
     "cellMetadata": {
      "byteLimit": 2048000,
      "rowLimit": 10000
     },
     "inputWidgets": {},
     "nuid": "c6968bd6-c3ba-4f17-b2c7-def7fab29139",
     "showTitle": false,
     "tableResultSettingsMap": {},
     "title": ""
    }
   },
   "source": [
    "# Fine Grained Demand Forecasting"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "application/vnd.databricks.v1+cell": {
     "cellMetadata": {
      "byteLimit": 2048000,
      "rowLimit": 10000
     },
     "inputWidgets": {},
     "nuid": "6d2f6ae6-53be-4572-9a19-b818034069ca",
     "showTitle": false,
     "tableResultSettingsMap": {},
     "title": ""
    }
   },
   "source": [
    "*Prerequisite: Make sure to run 01_Introduction_And_Setup before running this notebook.*\n",
    "\n",
    "In this notebook we use `AI_Forecast()` to forecast the SKU level demand\n",
    "\n",
    "Key highlights for this notebook:\n",
    "- `AI_Forecast()` is used to perform forecasting on time series data.   "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "application/vnd.databricks.v1+cell": {
     "cellMetadata": {
      "byteLimit": 2048000,
      "implicitDf": true,
      "rowLimit": 10000
     },
     "inputWidgets": {},
     "nuid": "eb6ba585-1bd1-4114-acde-322c10f260bc",
     "showTitle": false,
     "tableResultSettingsMap": {},
     "title": ""
    }
   },
   "outputs": [],
   "source": [
    "%sql\n",
    "USE CATALOG IDENTIFIER(:catalogName);\n",
    "USE SCHEMA IDENTIFIER(:dbName);"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "application/vnd.databricks.v1+cell": {
     "cellMetadata": {
      "byteLimit": 2048000,
      "rowLimit": 10000
     },
     "inputWidgets": {},
     "nuid": "aeefb34f-d634-4a92-bef8-9da779586614",
     "showTitle": false,
     "tableResultSettingsMap": {},
     "title": ""
    }
   },
   "source": [
    "\n",
    "## Using Databricks SQL AI Forecast Function\n",
    "`AI_Forecast()` is a table-valued function designed to extrapolate time series data into the future\n",
    "\n",
    "[AI_Forecast Reference](http://docs.databricks.com/aws/en/sql/language-manual/functions/ai_forecast)\n",
    "\n",
    "The `AI_Forecast()` function uses a forecasting procedure described as a *prophet-like piecewise linear and seasonal model*. This model incorporates seasonality and assumes a linear trend over time.\n",
    "\n",
    "At the moment of writing this it, is the only supported forecasting method currently available for this function.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "application/vnd.databricks.v1+cell": {
     "cellMetadata": {
      "byteLimit": 2048000,
      "rowLimit": 10000
     },
     "inputWidgets": {},
     "nuid": "0b920696-b36b-4799-9b90-62055a0b48e9",
     "showTitle": false,
     "tableResultSettingsMap": {},
     "title": ""
    }
   },
   "source": [
    "####Requirement\n",
    "- Pro or Serverless SQL warehouse\n",
    "- In Databricks Runtime 15.1 and above, this function is supported in Databricks notebooks, including notebooks that are run as a task in a Databricks workflow.\n",
    "- For batch inference workloads, Databricks recommends Databricks Runtime 15.4 ML LTS for improved performance."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "application/vnd.databricks.v1+cell": {
     "cellMetadata": {
      "byteLimit": 2048000,
      "rowLimit": 10000
     },
     "inputWidgets": {},
     "nuid": "6e91408d-ed75-4379-91eb-dea785bc0a6e",
     "showTitle": false,
     "tableResultSettingsMap": {},
     "title": ""
    }
   },
   "source": [
    "###Check AI_Forecast() Function Capabilities\n",
    "Using `AI_Forecast()` Function to forecast historical data for one SKU to show forecast quality for a specific SKU. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "application/vnd.databricks.v1+cell": {
     "cellMetadata": {
      "byteLimit": 2048000,
      "rowLimit": 10000
     },
     "inputWidgets": {},
     "nuid": "9ca1cffe-15bd-44d1-be12-ce03c0c7620d",
     "showTitle": false,
     "tableResultSettingsMap": {},
     "title": ""
    }
   },
   "outputs": [],
   "source": [
    "%sql\n",
    "select * from IDENTIFIER(:catalogName|| '.' || : dbName || '.part_level_demand' ) limit 10;"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "application/vnd.databricks.v1+cell": {
     "cellMetadata": {
      "byteLimit": 2048000,
      "implicitDf": true,
      "rowLimit": 10000
     },
     "inputWidgets": {},
     "nuid": "0a0935f8-c66c-4ba7-aeb4-779e95051e25",
     "showTitle": false,
     "tableResultSettingsMap": {},
     "title": ""
    }
   },
   "outputs": [
    {
     "output_type": "display_data",
     "data": {
      "text/plain": [
       "Databricks visualization. Run in Databricks to view."
      ]
     },
     "metadata": {
      "application/vnd.databricks.v1.subcommand+json": {
       "baseErrorDetails": null,
       "bindings": {},
       "collapsed": false,
       "command": "%sql WITH q AS (WITH input_data AS (\n  SELECT Date, Demand, SKU\n  FROM IDENTIFIER(:catalogName || '.' || :dbName || '.part_level_demand')\n  WHERE SKU = (\n    SELECT SKU\n    FROM IDENTIFIER(:catalogName || '.' || :dbName || '.part_level_demand')\n    LIMIT 1\n  )\n  AND Date <= (\n    SELECT date_add(MIN(Date), 365*2)\n    FROM IDENTIFIER(:catalogName || '.' || :dbName || '.part_level_demand')\n  )\n)\nSELECT\n  forecast.Date,\n  forecast.SKU,\n  demand.Demand,\n  forecast.Demand_Forecast\nFROM AI_FORECAST(\n    TABLE(input_data),\n    horizon => date_add(DAY, 40, (SELECT MAX(Date) FROM IDENTIFIER(:catalogName || '.' || :dbName || '.part_level_demand'))),\n    time_col => 'Date',\n    value_col => 'Demand',\n    group_col => 'SKU',\n    frequency => 'week',\n    parameters => '{ \"global_floor\": 0, \"yearly_order\": 12 }'\n) AS forecast\nLEFT JOIN IDENTIFIER(:catalogName || '.' || :dbName || '.part_level_demand') AS demand\n  ON forecast.Date = demand.Date AND forecast.SKU = demand.SKU\nORDER BY forecast.Date ASC) SELECT `Date`,SUM(`Demand`) `column_20467c69189`,SUM(`Demand_Forecast`) `column_20467c69192` FROM q GROUP BY `Date`",
       "commandTitle": "Line Plot",
       "commandType": "auto",
       "commandVersion": 0,
       "commentThread": [],
       "commentsVisible": false,
       "contentSha256Hex": null,
       "customPlotOptions": {
        "redashChart": [
         {
          "key": "type",
          "value": "CHART"
         },
         {
          "key": "options",
          "value": {
           "alignYAxesAtZero": true,
           "coefficient": 1,
           "columnConfigurationMap": {
            "x": {
             "column": "Date",
             "id": "column_20467c69186"
            },
            "y": [
             {
              "column": "Demand",
              "id": "column_20467c69189",
              "transform": "SUM"
             },
             {
              "column": "Demand_Forecast",
              "id": "column_20467c69192",
              "transform": "SUM"
             }
            ]
           },
           "dateTimeFormat": "DD/MM/YYYY HH:mm",
           "direction": {
            "type": "counterclockwise"
           },
           "error_y": {
            "type": "data",
            "visible": true
           },
           "globalSeriesType": "line",
           "isAggregationOn": true,
           "legend": {
            "traceorder": "normal"
           },
           "missingValuesAsZero": false,
           "numberFormat": "0,0.[00000]",
           "percentFormat": "0[.]00%",
           "series": {
            "error_y": {
             "type": "data",
             "visible": true
            },
            "stacking": null
           },
           "seriesOptions": {
            "column_20467c69189": {
             "name": "Real",
             "type": "line",
             "yAxis": 0
            },
            "column_20467c69192": {
             "name": "Forecasted",
             "type": "line",
             "yAxis": 0
            }
           },
           "showDataLabels": false,
           "sizemode": "diameter",
           "sortX": true,
           "sortY": true,
           "swappedAxes": false,
           "textFormat": "",
           "useAggregationsUi": true,
           "valuesOptions": {},
           "version": 2,
           "xAxis": {
            "labels": {
             "enabled": true
            },
            "type": "-"
           },
           "yAxis": [
            {
             "title": {
              "text": "Demand"
             },
             "type": "-"
            },
            {
             "opposite": true,
             "type": "-"
            }
           ]
          }
         }
        ]
       },
       "datasetPreviewNameToCmdIdMap": {},
       "diffDeletes": [],
       "diffInserts": [],
       "displayType": "redashChart",
       "error": null,
       "errorDetails": null,
       "errorSummary": null,
       "errorTraceType": null,
       "finishTime": 0,
       "globalVars": {},
       "guid": "",
       "height": "auto",
       "hideCommandCode": false,
       "hideCommandResult": false,
       "iPythonMetadata": null,
       "inputWidgets": {},
       "isLockedInExamMode": false,
       "latestAssumeRoleInfo": null,
       "latestUser": "a user",
       "latestUserId": null,
       "listResultMetadata": null,
       "metadata": {
        "byteLimit": 2048000,
        "implicitDf": true,
        "rowLimit": 10000
       },
       "nuid": "1c3d7dae-c8eb-4b4a-bbdb-a9d8ae0818e1",
       "origId": 0,
       "parentHierarchy": [],
       "pivotAggregation": null,
       "pivotColumns": null,
       "position": -1.0,
       "resultDbfsErrorMessage": null,
       "resultDbfsStatus": "INLINED_IN_TREE",
       "results": null,
       "showCommandTitle": false,
       "startTime": 0,
       "state": "input",
       "streamStates": {},
       "subcommandOptions": {
        "queryPlan": {
         "groups": [
          {
           "column": "Date",
           "type": "column"
          }
         ],
         "selects": [
          {
           "column": "Date",
           "type": "column"
          },
          {
           "alias": "column_20467c69189",
           "args": [
            {
             "column": "Demand",
             "type": "column"
            }
           ],
           "function": "SUM",
           "type": "function"
          },
          {
           "alias": "column_20467c69192",
           "args": [
            {
             "column": "Demand_Forecast",
             "type": "column"
            }
           ],
           "function": "SUM",
           "type": "function"
          }
         ]
        }
       },
       "submitTime": 0,
       "subtype": "tableResultSubCmd.visualization",
       "tableResultIndex": 0,
       "tableResultSettingsMap": {},
       "useConsistentColors": false,
       "version": "CommandV1",
       "width": "auto",
       "workflows": null,
       "xColumns": null,
       "yColumns": null
      }
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "%sql\n",
    "WITH input_data AS (\n",
<<<<<<< Updated upstream
    "  SELECT Date, Demand, SKU FROM part_level_demand\n",
    "  WHERE SKU = (SELECT SKU FROM part_level_demand LIMIT 1) \n",
    "  AND Date <= (SELECT DATE_ADD(MIN(Date), 365*2) FROM part_level_demand)\n",
=======
    "  SELECT Date, Demand, SKU\n",
    "  FROM IDENTIFIER(:catalogName || '.' || :dbName || '.part_level_demand')\n",
    "  WHERE SKU = (\n",
    "    SELECT SKU\n",
    "    FROM IDENTIFIER(:catalogName || '.' || :dbName || '.part_level_demand')\n",
    "    LIMIT 1\n",
    "  )\n",
    "  AND Date <= (\n",
    "    SELECT date_add(MIN(Date), 365*2)\n",
    "    FROM IDENTIFIER(:catalogName || '.' || :dbName || '.part_level_demand')\n",
    "  )\n",
>>>>>>> Stashed changes
    ")\n",
    "SELECT\n",
    "  forecast.Date,\n",
    "  forecast.SKU,\n",
    "  demand.Demand,\n",
    "  forecast.Demand_Forecast\n",
    "FROM AI_FORECAST(\n",
    "    TABLE(input_data),\n",
<<<<<<< Updated upstream
    "    horizon => DATE_ADD(DAY, 40, (SELECT MAX(Date) FROM part_level_demand)) ,\n",
=======
    "    horizon => date_add(DAY, 40, (SELECT MAX(Date) FROM IDENTIFIER(:catalogName || '.' || :dbName || '.part_level_demand'))),\n",
>>>>>>> Stashed changes
    "    time_col => 'Date',\n",
    "    value_col => 'Demand',\n",
    "    group_col => 'SKU',\n",
    "    frequency => 'week',\n",
    "    parameters => '{ \"global_floor\": 0, \"yearly_order\": 12 }'\n",
    ") AS forecast\n",
<<<<<<< Updated upstream
    "LEFT JOIN part_level_demand AS demand\n",
    "ON forecast.Date = Demand.Date AND forecast.SKU = demand.SKU \n",
    "ORDER BY Date ASC"
=======
    "LEFT JOIN IDENTIFIER(:catalogName || '.' || :dbName || '.part_level_demand') AS demand\n",
    "  ON forecast.Date = demand.Date AND forecast.SKU = demand.SKU\n",
    "ORDER BY forecast.Date ASC"
>>>>>>> Stashed changes
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "application/vnd.databricks.v1+cell": {
     "cellMetadata": {
      "byteLimit": 2048000,
      "rowLimit": 10000
     },
     "inputWidgets": {},
     "nuid": "6488f8a2-8638-477b-826f-adebf14255b6",
     "showTitle": false,
     "tableResultSettingsMap": {},
     "title": ""
    }
   },
   "source": [
    "## Perform demand forecasts using AI_Forecast() function\n",
    "\n",
    "Apply the `AI_Forecast()` function to our whole dataset to generate demand forecasts for the next 40 days and store the results into a delta table."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "application/vnd.databricks.v1+cell": {
     "cellMetadata": {
      "byteLimit": 2048000,
      "implicitDf": true,
      "rowLimit": 10000
     },
     "inputWidgets": {},
     "nuid": "651a684e-da2f-4183-97c3-4fa9bd57d558",
     "showTitle": false,
     "tableResultSettingsMap": {},
     "title": ""
    }
   },
   "outputs": [],
   "source": [
    "%sql\n",
<<<<<<< Updated upstream
    "CREATE OR REPLACE TABLE part_level_demand_with_forecasts AS (\n",
    "SELECT forecast.Date, forecast.SKU, demand.Product, forecast.Demand_Forecast AS Demand, TRUE AS is_forecast FROM AI_FORECAST(\n",
    "    TABLE(SELECT Date, Demand, SKU FROM part_level_demand),\n",
    "    horizon => DATE_ADD(DAY, 90, (SELECT MAX(Date) FROM part_level_demand)),\n",
=======
    "CREATE OR REPLACE TABLE \n",
    "  IDENTIFIER(:catalogName || '.' || :dbName || '.part_level_demand_with_forecasts')\n",
    "AS (\n",
    "  SELECT\n",
    "    forecast.Date,\n",
    "    forecast.SKU,\n",
    "    demand.Product,\n",
    "    forecast.Demand_Forecast AS Demand,\n",
    "    TRUE AS is_forecast\n",
    "  FROM AI_FORECAST(\n",
    "    TABLE(\n",
    "      SELECT\n",
    "        Date,\n",
    "        Demand,\n",
    "        SKU\n",
    "      FROM IDENTIFIER(:catalogName || '.' || :dbName || '.part_level_demand')\n",
    "    ),\n",
    "    horizon => DATE_ADD(\n",
    "      DAY,\n",
    "      90,\n",
    "      (SELECT MAX(Date) FROM IDENTIFIER(:catalogName || '.' || :dbName || '.part_level_demand'))\n",
    "    ),\n",
>>>>>>> Stashed changes
    "    time_col => 'Date',\n",
    "    value_col => 'Demand',\n",
    "    group_col => 'SKU',\n",
    "    frequency => 'week',\n",
    "    parameters => '{ \"global_floor\": 0, \"yearly_order\": 12 }'\n",
<<<<<<< Updated upstream
    ") as forecast\n",
    "LEFT JOIN (SELECT DISTINCT SKU, Product FROM part_level_demand) AS demand\n",
    "ON forecast.SKU = demand.SKU\n",
    "UNION ALL \n",
    "SELECT Date, SKU, Product, Demand, FALSE as is_forecast FROM part_level_demand\n",
    "ORDER BY SKU, Date\n",
    ")\n"
=======
    "  ) AS forecast\n",
    "  LEFT JOIN (\n",
    "    SELECT DISTINCT\n",
    "      SKU,\n",
    "      Product\n",
    "    FROM IDENTIFIER(:catalogName || '.' || :dbName || '.part_level_demand')\n",
    "  ) AS demand\n",
    "    ON forecast.SKU = demand.SKU\n",
    "\n",
    "  UNION ALL\n",
    "\n",
    "  SELECT\n",
    "    Date,\n",
    "    SKU,\n",
    "    Product,\n",
    "    Demand,\n",
    "    FALSE AS is_forecast\n",
    "  FROM IDENTIFIER(:catalogName || '.' || :dbName || '.part_level_demand')\n",
    "\n",
    "  ORDER BY SKU, Date\n",
    ")"
>>>>>>> Stashed changes
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "application/vnd.databricks.v1+cell": {
     "cellMetadata": {
      "byteLimit": 2048000,
      "implicitDf": true,
      "rowLimit": 10000
     },
     "inputWidgets": {},
     "nuid": "d41b94d4-3e27-40e5-9d14-cd405b32db02",
     "showTitle": false,
     "tableResultSettingsMap": {},
     "title": ""
    }
   },
   "outputs": [],
   "source": [
    "%sql\n",
<<<<<<< Updated upstream
    "SELECT * FROM part_level_demand_with_forecasts;"
=======
    "SELECT * FROM IDENTIFIER(:catalogName || '.' || :dbName || '.part_level_demand_with_forecasts')"
>>>>>>> Stashed changes
   ]
  }
 ],
 "metadata": {
  "application/vnd.databricks.v1+notebook": {
   "computePreferences": {
    "hardware": {
     "accelerator": null,
     "gpuPoolId": null,
     "memory": null
    }
   },
   "dashboards": [],
   "environmentMetadata": null,
   "inputWidgetPreferences": null,
   "language": "python",
   "notebookMetadata": {
    "mostRecentlyExecutedCommandWithImplicitDF": {
     "commandId": -1,
     "dataframes": [
      "_sqldf"
     ]
    },
    "pythonIndentUnit": 2
   },
   "notebookName": "02_Fine_Grained_Demand_Forecasting_AI_Forecast",
   "widgets": {
    "catalogName": {
     "currentValue": "maxkoehler_demos",
     "nuid": "0e498a46-9745-49a8-b4df-cf6e9b651ab1",
     "typedWidgetInfo": {
      "autoCreated": false,
      "defaultValue": "maxkoehler_demos",
      "label": "Catalog Name",
      "name": "catalogName",
      "options": {
       "widgetDisplayType": "Text",
       "validationRegex": null
      },
      "parameterDataType": "String"
     },
     "widgetInfo": {
      "widgetType": "text",
      "defaultValue": "maxkoehler_demos",
      "label": "Catalog Name",
      "name": "catalogName",
      "options": {
       "widgetType": "text",
       "autoCreated": null,
       "validationRegex": null
      }
     }
    },
    "dbName": {
     "currentValue": "demand_db",
     "nuid": "f71a5129-0151-44f9-8cf7-d6428fed01c3",
     "typedWidgetInfo": {
      "autoCreated": false,
      "defaultValue": "demand_db",
      "label": "Database Name",
      "name": "dbName",
      "options": {
       "widgetDisplayType": "Text",
       "validationRegex": null
      },
      "parameterDataType": "String"
     },
     "widgetInfo": {
      "widgetType": "text",
      "defaultValue": "demand_db",
      "label": "Database Name",
      "name": "dbName",
      "options": {
       "widgetType": "text",
       "autoCreated": null,
       "validationRegex": null
      }
     }
    },
    "reset_all_data": {
     "currentValue": "false",
     "nuid": "9e183d2c-2379-4a44-91a5-b35afe54b8f0",
     "typedWidgetInfo": {
      "autoCreated": false,
      "defaultValue": "false",
      "label": "Reset all data",
      "name": "reset_all_data",
      "options": {
       "widgetDisplayType": "Dropdown",
       "choices": [
        "true",
        "false"
       ],
       "fixedDomain": true,
       "multiselect": false
      },
      "parameterDataType": "String"
     },
     "widgetInfo": {
      "widgetType": "dropdown",
      "defaultValue": "false",
      "label": "Reset all data",
      "name": "reset_all_data",
      "options": {
       "widgetType": "dropdown",
       "autoCreated": null,
       "choices": [
        "true",
        "false"
       ]
      }
     }
    }
   }
  },
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 0
}
