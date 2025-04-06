##################################################
SCREENER - CON PROXIES
##################################################
##	IMPORTAR LIBRERIAS

#!pip install ta yahooquery

from yahooquery import Ticker
import pandas as pd
import datetime as dt
from numpy import where
from datetime import datetime
import ta
import requests, re
from bs4 import BeautifulSoup

import warnings
warnings.filterwarnings("ignore")

##################################################
##	GET PROXIES

regex = r"[0-9]+(?:\.[0-9]+){3}:[0-9]+"
c = requests.get("https://spys.me/proxy.txt")
test_str = c.text
a = re.finditer(regex, test_str, re.MULTILINE)
with open("proxies_list.txt", 'w') as file:
    for i in a:
       print(i.group(),file=file)

d = requests.get("https://free-proxy-list.net/")
soup = BeautifulSoup(d.content, 'html.parser')
td_elements = soup.select('.fpl-list .table tbody tr td')
ips = []
ports = []
for j in range(0, len(td_elements), 8):
    ips.append(td_elements[j].text.strip())
    ports.append(td_elements[j + 1].text.strip())
with open("proxies_list.txt", "a") as myfile:
    for ip, port in zip(ips, ports):
        proxy = f"{ip}:{port}"
        print(proxy, file=myfile)


##################################################
proxy_list = [31, 15, 12, 7, 65, 23, 10, 85]

tickers_validated = list(pd.read_excel('tickers_validated_full.xlsx')['symbol'])

ticker_list = [tickers_validated[i:i+1000] for i in range(0, len(tickers_validated), 1000)]

df_data_list = []

for idx in range(len(ticker_list)):
  tickers_str = " ".join(ticker_list[idx])
  tickers_prueba = Ticker(tickers_str, asynchronous=True,
                         proxies = {'http': 'http://'+ ips[idx] + ':' + ports[idx]})
  try:
    df_prueba1 = tickers_prueba.history(period="200d", interval="1h")
    df_data_list.append(df_prueba1.copy())
  except:
    df_prueba1 = None
    print('error en batch', idx)
  
####### clean data to yfinance output
df_full = pd.concat(df_data_list,axis=0).reset_index().copy()

list_ticker = df_full['symbol'].unique()

# Usar groupby para evitar filtrar dentro del bucle
grouped = df_full.groupby('symbol')

# Lista para almacenar DataFrames
df_list = []

for symbol, df_temp in grouped:
    df_temp = df_temp.set_index('date').copy()
    df_temp = df_temp.rename(columns=lambda x: x.capitalize())
    df_temp.columns = pd.MultiIndex.from_tuples([(symbol, col) for col in df_temp.columns])
    df_temp = df_temp.drop_duplicates()
    df_list.append(df_temp)

#df_list = [df[~df.index.duplicated(keep='first')] for df in df_list]
df_list = [df.drop_duplicates() for df in df_list]

# Concatenar todos los DataFrames al final
temp_df = pd.concat(df_list, axis=1)
 
### strategy 1
end = datetime.today()
end_utc = pd.Timestamp(end).tz_localize("UTC")
start = end - pd.Timedelta(days=200)


# Desactivar warnings de pandas
warnings.simplefilter("ignore", FutureWarning)
#tickers_symbols = all_tickers[1:11]

"""parte diego actualizada"""

from datetime import datetime, timedelta
df_data = temp_df.copy()

signals = []
for ticker in list_ticker:
    try:
        df = df_data[ticker]
        df.dropna(inplace=True)


        df['MA100'] = df['Close'].rolling(window=100).mean()
        df['Prev_Close'] = df['Close'].shift(1)
        df['Prev_MA100'] = df['MA100'].shift(1)

        df['last_down'] = df['Close'].tail(1) < df['MA100'].tail(1)
        df['Cross_Down'] = (df['Prev_Close'] < df['Prev_MA100']) & (df['Close'] < df['MA100'] )
        df['Monthly_Trend'] = (df['Close'] > df['MA100'].shift(130)) & (df['Close'] > df['MA100'].shift(60))

        recent_signals = df[(df['Cross_Down']) & (df['Monthly_Trend']) & (df['last_down'])]
        #recent_signals = recent_signals[recent_signals.index > (end - timedelta(days=7))]
        recent_signals = recent_signals[recent_signals.index > (end_utc  - timedelta(days=7))]

        if not recent_signals.empty:
            signals.append((ticker, recent_signals.index[-1]))

    except Exception as e:
        print(f"error con {ticker}: {e}")

signals_1 = signals.copy()
#signals = signals[:30]

codigos = [t[0] for t in signals_1]

filtered_tic = [tic for tic in codigos if df_data[tic]['Close'].isnull().sum() < 5]

# Exportar a CSV
tick = pd.DataFrame(filtered_tic, columns=["Symbol"])
tick.to_csv("symbols.csv", index=False)

##Add watchlist
#crear variable date de la fecha de hoy
tick['date'] = datetime.today().strftime('%Y-%m-%d')
tick = tick[['date','Symbol']]

# Exportar a watchlist.csv sin sobrescribir si el archivo ya existe
import os

file_path = "watchlist1.csv"
if os.path.exists(file_path):
    tick.to_csv(file_path, mode='a', header=False, index=False)  # Agregar sin sobrescribir encabezados
else:
    tick.to_csv(file_path, mode='w', index=False)  # Crear el archivo con encabezados si no existe
