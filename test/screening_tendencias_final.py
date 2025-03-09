###Screening
# OJO: Este screening solo analizar los tickers de S&P500, Dow Jones y Russell 2000
# Suman alrededor de 2301 tickers
# Si se incluyen los tickers de NASDAQ, la cantidad de tickers suben a mas de 9000, pero
# yfinance no descarga todos los datos y el tiempo de demora del codigo aumenta de 8-10 min a 30 min.

# pip install ffn
#!pip install ta
# !pip install pmdarima

#import pandas_datareader.data as web
import datetime as dt
from numpy import where
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
#import ffn
#from sklearn.cluster import KMeans
#from sklearn.preprocessing import MinMaxScaler
#from sklearn.preprocessing import StandardScaler

#import seaborn as sns
import pandas as pd
import numpy as np

import ta

import warnings
warnings.filterwarnings("ignore")

end = datetime.today()
start = end - pd.Timedelta(days=200)

# datos de S&P 500
payload = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
df_sp500 = payload[0]
df_sp500["Date added"] = pd.to_datetime(df_sp500["Date added"])
df_sp500 = df_sp500[df_sp500["Date added"] > pd.Timestamp("2000-01-01")]

# tickers y sectores del S&P 500
sp500_tickers = df_sp500['Symbol'].values.tolist()
sp500_sectors = df_sp500['GICS Sector'].values.tolist()

# S&P 500
sector_dict = dict(zip(sp500_tickers, sp500_sectors))

# Dow Jones, NASDAQ y Russell 2000
dow_jones = {
    'Ticker': ['MMM', 'AXP', 'AAPL', 'BA', 'CAT', 'CVX', 'CSCO', 'KO', 'DIS', 'DWDP',
                'XOM', 'GS', 'HD', 'IBM', 'INTC', 'JNJ', 'JPM', 'MCD', 'MEK', 'MSFT',
                'NKE', 'PFE', 'PG', 'TRV', 'UTX', 'UNH', 'VZ', 'V', 'WMT', 'WBA'],
    'Sector': ['Industrials', 'Financials', 'Technology', 'Industrials', 'Industrials',
               'Energy', 'Technology', 'Consumer Staples', 'Communication Services', 'Materials',
                'Energy', 'Financials', 'Consumer Discretionary', 'Technology', 'Technology',
                'Health Care', 'Financials', 'Consumer Discretionary', 'Health Care', 'Technology',
                'Consumer Discretionary', 'Health Care', 'Consumer Staples', 'Financials', 'Industrials',
                'Health Care', 'Communication Services', 'Financials', 'Consumer Discretionary',
                'Consumer Discretionary']
}
dow_jones_df = pd.DataFrame(dow_jones)
dow_jones_tickers = dow_jones_df['Ticker'].values.tolist()
dow_jones_sectors = dow_jones_df['Sector'].values.tolist()

# NASDAQ y Russell 2000


russell_2000 = pd.read_excel("russell_2000.xlsx")
russell_2000_tickers = russell_2000['Ticker'].values.tolist()
russell_2000_sectors = russell_2000['Sector'].values.tolist()

# todas las listas de tickers y sectores
#all_tickers = sp500_tickers + dow_jones_tickers + nasdaq_tickers + russell_2000_tickers
all_tickers = sp500_tickers + dow_jones_tickers +  russell_2000_tickers
#all_sectors = sp500_sectors + dow_jones_sectors + nasdaq_sectors + russell_2000_sectors
all_sectors = sp500_sectors + dow_jones_sectors + russell_2000_sectors

all_tickers = [ticker for ticker in all_tickers if pd.notna(ticker)]
# all_sectors = [sector for sector in all_sectors if pd.notna(sector)]
# all_sectors = [sector if pd.notna(sector) else "Unknown" for sector in all_sectors]

sector_dict.update(dict(zip(all_tickers, all_sectors)))

# empty frame
df_final1 = pd.DataFrame(columns=["ticker", "sector", "flag_retroceso"], index=range(len(all_tickers)))

tickers_symbols = all_tickers#[1:10]

end = datetime.today()
end_utc = pd.Timestamp(end).tz_localize("UTC")
start = end - pd.Timedelta(days=200)

import time
start_time = time.time()
df_data1 = yf.download(tickers=tickers_symbols, interval="60m", start=start, end=end, group_by='ticker')
end_time = time.time()

execution_time = end_time - start_time
print(f"Tiempo de ejecución: {execution_time:.2f} segundos")

###########

"""filters"""

#display(df_data1.info)
#display(df_data1.head())
#display(df_data1.isnull().sum())


# ##########

from datetime import timedelta
from datetime import datetime, timedelta

df_data = df_data1.copy()

signals = []
for ticker in tickers_symbols:
    try:
        df = df_data[ticker]
        df.dropna(inplace=True)

        if ((df['Close'] > 20).any()):
        #and (df['Volume'].tail(7).mean() > df['Volume'].rolling(window=30).mean().iloc[-1]):


            df['MA100'] = df['Close'].rolling(window=100).mean()
            df['Prev_Close'] = df['Close'].shift(1)
            df['Prev_MA100'] = df['MA100'].shift(1)
            df['RSI'] = ta.momentum.rsi(close=df['Close'], window=20)
            df['RSI_oversold_last5'] = (df['RSI'].rolling(window=20, min_periods=20).min() < 35)
            df['last_down'] = df['Close'].tail(1) < df['MA100'].tail(1)
            df['Cross_Down'] = (df['Prev_Close'] < df['Prev_MA100']) & (df['Close'] < df['MA100'] )
            df['Monthly_Trend'] = (df['Close'] > df['MA100'].shift(200)) & (df['Close'] > df['MA100'].shift(100))

            recent_signals = df[(df['Cross_Down']) & (df['Monthly_Trend']) & (df['last_down']) & (df['RSI_oversold_last5'])]
            recent_signals = recent_signals[recent_signals.index > (end_utc  - timedelta(days=1))]
            #recent_signals = recent_signals[recent_signals.index > (end - timedelta(days=1))]

            if not recent_signals.empty:
                signals.append((ticker, recent_signals.index[-1]))


    except Exception as e:
        print(f"Error con {ticker}: {e}")

signals_1 = signals.copy()
#signals = signals[:30]

codigos = [t[0] for t in signals_1]
#codigos.append(execution_time)

#print(codigos)

# Lista de tickers: codigos
# Crear un DataFrame con el encabezado "Symbols"
tick = pd.DataFrame(codigos, columns=["Symbol"])

# Exportar a CSV
tick.to_csv("symbols.csv", index=False)