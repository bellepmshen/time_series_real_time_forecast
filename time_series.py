import itertools
import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

class Time_Series:
    
    def __init__(self) -> None:
        pass
    
    def get_data(self, ticker_symbol):
        """This function is for loading the 7-day of stock price by minute
        
        Parameters
        ----------
        ticker_symbol : str
            the ticker symbol from a specific company
            
        Returns
        -------
        pandas dataframe
            a dataframe contains stock price
        """
        try:
            
            # get stock price:
            data = yf.download(tickers = ticker_symbol, period = "7d", interval = "1m", 
                               progress = False
                               )
            # print out the latest timestamp of the data:
            print(f"The last timestamp of the data (New York timezone): {data.index[-1].strftime('%Y-%m-%d %H:%M:%S')}")
            
            self.data = data
            return self.data
        
        except Exception as err:
            print(err)       
    
    def split(self):
        """This function is for train & test split for time series data

        Returns
        -------
        series
            train & test series
        """
        try:
            train = self.data[['Open']].iloc[:-50]
            test = self.data[['Open']].iloc[-50:]
            
            self.train = train
            self.test = test
            
            return self.train, self.test
        
        except Exception as err:
            print(err)
            
    def make_log(self):
        """This function is for taking log1p transform on stock price.

        Returns
        -------
        train_1, test_1
            the train & test after taking log1p transform
        """
        try:
            train_1 = np.log1p(self.train)
            test_1 = np.log1p(self.test)
            
            self.train_1 = train_1
            self.test_1 = test_1

            return self.train_1, self.test_1
        
        except Exception as err:
            print(err)    

    def generate_params(self):
        """This function is for generating P, D, Q permutation for ARIMA model.

        Returns
        -------
        params, list
            a list of each P, D, Q combination
        """
        try:
            p = range(1, 7)
            d = range(1, 3)
            q = range(3)
            params = list(itertools.product(p, d, q))
            
            self.params = params
            
            return self.params
        
        except Exception as err:
            print(err)   
            
    def training(self, i, xtrain, xtest):
        """This function is for ARIMA model training.

        Parameters
        ----------
        i : tuple
            the (P, D, Q) combination
        xtrain : pandas series
            the training series 
        xtest : pandas series
            the test series

        Returns
        -------
        prediction, list
            a list of prediction
        """
        try:
            prediction = []
            
            for j in range(len(xtest)):
                try:
                    model = ARIMA(xtrain, order = i).fit()
                    pred = model.forecast()[0]
                    prediction.append(pred)
                    xtrain = xtrain.append(pd.Series(xtest.iloc[j]), ignore_index = True)
                    
                except:
                    prediction.append(np.nan)
                    
            return prediction
        
        except Exception as err:
            print(err) 
    
    def rmse_calculation(self):
        """This function is for ARIMA model training & get a list of RMSE of each
        permutation of (P, D, Q)

        Returns
        -------
        rmse_box, list
            a list of RMSE from ARIMA model training with each (P, D, Q) permutation
        best_pdq, tuple
            a tuple of the best (P, D, Q)
        """
        try:
            rmse_box = []
            for i in self.params[:1]:
                try:
                    train_2 = self.train_1.copy()
                    prediction = self.training(i, train_2, self.test_1)
                    rmse = np.sqrt(
                        mean_squared_error(np.expm1(self.test_1), np.expm1(prediction))
                        )
                    rmse_box.append(rmse)
                    # print(f"(P, D, Q): {i}, RMSE: {rmse}")
                except:
                    rmse_box.append(np.nan)
                    # print(f"(P, D, Q): {i}, RMSE: nan")
            
            best_pdq = self.params[rmse_box.index(min(rmse_box))]
            # print(f"smallest RMSE: {min(rmse_box)}, \
            # (P, D, Q) = {best_pdq}.") 
            
            self.rmse_box = rmse_box
            self.best_pdq = best_pdq
            
            return self.rmse_box, self.best_pdq
        
        except Exception as err:
            print(err) 
    
    def final_training(self):
        """This function is to get the final model and make a prediction for 
        the next minute open price.

        Returns
        -------
        str
            a message to show the next minute open price
        """
        try:
            final_model = ARIMA(self.data[['Open']], order = self.best_pdq).fit()
            pred = final_model.forecast()[0]
            next_open_price = float(np.round_(pred, 2))
            
            return f"The open price will be ${next_open_price} in the next minute."
        
        except Exception as err:
            print(err)     
