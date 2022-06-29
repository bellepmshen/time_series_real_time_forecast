# Time Series - Real Time Forecast

Inside this repo, there are 2 files for obtaining the next minute forecast.

1. time_series.py: This script contains the functions for getting the next minute forecast.
2. forecast_uni.ipynb: This notebook is to run the whole functions from time_series.py and get the next minute forecast.

## instructions to get the forecast on the next minute:

1. Download the 2 files, time_series.py and forecast_uni.ipynb and run them in the same directory.
2. Changing ticker symbol:

    Inside forecast_uni.ipynb, change the argument in the "get_data()" function.

    ex: I want to get the forecast of Apple Inc. and the ticker symbol of Apple Inc. is "AAPL." Then, the code will be:

        stock = ts.get_data('aapl')



