import pandas as pd
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from src.data_preprocessing import *

def cal_ex_moving_avg(df,EMA_window):
    """"
    This function provides the info about Exponential moving average
    
    Args : EMA_window as number of days
    Returns: A dataframe of EMA
    """
    # Calculate EMA using pandas
    ema = df['Close'].ewm(span=EMA_window, adjust=False).mean()

    # Add EMA to DataFrame
    df['EMA'] = ema
    return df




def calculate_rsi(data,window_days):
    """This function will take parameters data and window_days 
    and calculate the RSI for the stock 
    Args : data, window_days
    
    Returns:
    RSI
    """
    # Calculate daily price changes
    data['Price_Change'] = data['Close'].diff()

    # Calculate gains and losses
    data['Gain'] = data['Price_Change'].apply(lambda x: x if x > 0 else 0)
    data['Loss'] = data['Price_Change'].apply(lambda x: abs(x) if x < 0 else 0)

    # Calculate average gains and losses over a 5-day period
 
    data['Avg_Gain'] = data['Gain'].rolling(window=window_days).mean()
    data['Avg_Loss'] = data['Loss'].rolling(window=window_days).mean()

    # Calculate Relative Strength (RS) & RSI
    data['RS'] = data['Avg_Gain'] / data['Avg_Loss']
    data['RSI'] = 100 - (100 / (1 + data['RS']))
    data.dropna(inplace=True)
    return data
