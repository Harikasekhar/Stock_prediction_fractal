import yfinance as yf
import pandas as pd


def get_stock_data(stock_name,period):
    """
    Function to collect stock data from Yahoo Finance.

    Args:
    ticker (str): Ticker symbol of the stock (e.g., "TSLA" for Tesla).
    start_date (str): Start date in "YYYY-MM-DD" format.
    end_date (str): End date in "YYYY-MM-DD" format.

    Returns:
    DataFrame: Pandas DataFrame containing the stock data.
    """
    try:
        print(stock_name,period)
        stock_ticker = yf.Ticker(stock_name)
        history = stock_ticker.history(period=period)
        df = pd.DataFrame(history)
        df.reset_index(inplace=True) # to reset index and convert it to column
        return df

    except Exception as e:
        print("An error occurred:", str(e))
        return None


# df = get_stock_data('TSLA','1mo')
# print(df.head(5))