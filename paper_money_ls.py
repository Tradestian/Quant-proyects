#%%
import asyncio
import pandas as pd   
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns  
import ta  
import warnings  
from datetime import datetime, timedelta, time  
from pytz import timezone  
from concurrent.futures import ThreadPoolExecutor  
from ib_insync import IB, Stock, MarketOrder, util  
from ib_insync import LimitOrder, StopOrder 
import yfinance as yf  
import pandas_datareader.data as web  
from ib_insync import MarketOrder

warnings.filterwarnings("ignore")  

util.startLoop()

ib = IB()

ib.connect('127.0.0.1', 4002, clientId=93456)

df = pd.read_csv('watchlist1.csv')

entrada = 1000
API_SLEEP = 10
INTERVAL = 900
DURATION = '40 D'
BAR_SIZE = '30 mins'
SYMBOLS = df['Symbol'].iloc[-100:].unique().tolist()


MAX_CONCURRENT_REQUESTS = 10
MARKET_CLOSE = time(15, 30) # 14 (diferencia de 1 hora entre la hora de Perú y Canadá)
MARKET_OPEN = time(9, 30)  # 8 (diferencia de 1 hora entre la hora de Perú y Canadá)
qualified_stocks = [Stock(symbol=symbol, exchange='SMART', currency='USD') for symbol in SYMBOLS]
historical_data = {}

semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)


orders_df = pd.DataFrame([], columns=['OrderId', 'Symbol', 'Action', 'Quantity', 'Price', 'Status', 'Take Profit', 'Stop Loss', 'Date'])

async def wait_for_market_open(): 
    print("Esperando la apertura del mercado...")
    while True:
        current_time = datetime.now().time()
        if current_time >= MARKET_OPEN:
            print("¡El mercado está abierto! Iniciando operaciones...")
            break
        await asyncio.sleep(10)

async def fetch_data(stock): 
    async with semaphore:
        try:
            bars = await ib.reqHistoricalDataAsync(
                stock,
                endDateTime="",
                durationStr=DURATION,
                barSizeSetting=BAR_SIZE,
                whatToShow='MIDPOINT',
                useRTH=False
            )
            if bars:
                df = util.df(bars)
                df['symbol'] = stock.symbol
                df['date'] = pd.to_datetime(df['date'])
                return stock.symbol, df
            else:
                return stock.symbol, None
        except Exception as e:
            print(f"Error con {stock.symbol}: {e}")
            return stock.symbol, None

async def get_historical_data(): 
    tasks = [fetch_data(stock) for stock in qualified_stocks]
    results = await asyncio.gather(*tasks)
    for symbol, data in results:
        if data is not None:
            historical_data[symbol] = data 

def process_data(): 
    if not historical_data:
        print("No hay datos para procesar.")
        return pd.DataFrame()

    combined_data = pd.concat(historical_data.values(), ignore_index=True)
    combined_data = combined_data.drop_duplicates(subset=['date', 'symbol'])
    pivot_df = combined_data.pivot(index='date', columns='symbol', values=['open', 'high', 'low', 'close', 'volume'])
    pivot_df = pivot_df.swaplevel(axis=1).sort_index(axis=1)
    return pivot_df

async def execute_signals(symbols, combined_data): 
    tasks = []
    for symbol in symbols:
        if symbol not in combined_data.columns.get_level_values(0):
            continue
        df = combined_data[symbol].copy()
        if not {'close', 'high', 'low', 'volume'}.issubset(df.columns):
            continue

        df['RSI'] = ta.momentum.rsi(close=df['close'], window=20)
        df['SMA100'] = df['close'].rolling(window=100, min_periods=100).mean()

        df['RSI_oversold_last5'] = (df['RSI'].rolling(window=20, min_periods=10).min() < 35).astype(int)
        df['RSI_overbught_last5'] = (df['RSI'].rolling(window=20, min_periods=10).max() > 65).astype(int)

        # Rango máximo-mínimo
        df['max_min_range'] = df['high'].rolling(window=20, min_periods=20).max() - df['low'].rolling(window=20, min_periods=20).min()

        # Señal de ruptura de SMA100
        df['break_up_SMA'] = (
            (df['close'] > df['SMA100']) &
            # (df['max_min_range'] <= df['max_min_range'].mean()) &
            (df['close'].shift(1) <= df['SMA100'].shift(1)) &
            (df['close'].shift(2) <= df['SMA100'].shift(2)) 
        ).astype(int)

        df['break_down_SMA'] = (
            (df['close'] < df['SMA100']) &         
            # (df['max_min_range'] <= df['max_min_range'].mean()) & # Es solo rango para limitar la volatilidad
            (df['close'].shift(1) >= df['SMA100'].shift(1)) &
            (df['close'].shift(2) >= df['SMA100'].shift(2))
        ).astype(int)
        
        df['buy_signal'] = (
                    (df['break_up_SMA']==1) &  
                    (df['RSI_oversold_last5'] == 1)).astype(int)
        
        df['sell_signal'] = (
                    (df['break_down_SMA']==1) &  
                    (df['RSI_overbught_last5'] == 1)).astype(int) 

        buy_signals_count = df['buy_signal'].iloc[-5:].sum() 
        print(f"Señales de compra generadas para {symbol}: {buy_signals_count}")
        
        sell_signals_count = df['sell_signal'].iloc[-5:].sum()
        print(f"Señales de venta generadas para {symbol}: {sell_signals_count}")

        if orders_df[orders_df['Symbol'] == symbol]['Status'].isin(['Pending', 'Filled', 'Done']).any():
            print(f"Ya existe una orden para {symbol}. No se generará otra.")
            continue  

        if df['buy_signal'].iloc[-1]:  
            
            qty = entrada // df['close'].iloc[-1]
            if qty <= 0:
                print(f"Cantidad calculada para {symbol} es 0. Saltando operación.")
                continue
            contract = Stock(symbol, 'SMART', 'USD')
            ib.qualifyContracts(contract)

            parent_order = MarketOrder('BUY', qty)  
            trade = ib.placeOrder(contract, parent_order)

            # Verificar órdenes abiertas en IBKR para depuración
            print(f"Órdenes abiertas en IBKR: {ib.openOrders()}")
            print(f"Orden enviada para {symbol}, esperando ejecución...")

            order_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # wait_for_order_fill(trade, contract, parent_order)
            # fill_price = trade.fills[0].price if trade.fills else None
            # if fill_price:
            entry_price = df['close'].iloc[-1]
            take_profit_price = round(entry_price * 1.05, 2)
            stop_loss_price = round(entry_price * 0.978, 2)

            takeProfit = LimitOrder('SELL', qty, take_profit_price, tif='GTC',parentId=parent_order.orderId, transmit=True)
            stopLoss = StopOrder('SELL', qty, stop_loss_price, tif='GTC', parentId=parent_order.orderId, transmit=True)

            tp_trade = ib.placeOrder(contract, takeProfit)
            sl_trade = ib.placeOrder(contract, stopLoss)
            # ib.sleep(1)
            # assert tp_trade.orderStatus.status == 'Submitted'
            # assert tp_trade in ib.openTrades()
            # assert sl_trade.orderStatus.status == 'Submitted'
            # assert sl_trade in ib.openTrades()
            orders_df.loc[len(orders_df)] = [parent_order.orderId, symbol, 'LONG', qty, entry_price,
                                                'Done', take_profit_price, stop_loss_price, order_date]
            print(f"Orden de TP enviada a {take_profit_price} y SL enviada a {stop_loss_price} para {symbol}.")
            # else:
            #     print(f"No se pudo obtener el precio de llenado para {symbol}.")
        elif df['sell_signal'].iloc[-1]:

            qty = entrada // df['close'].iloc[-1]
            if qty <= 0:
                print(f"Cantidad calculada para {symbol} es 0. Saltando operación.")
                continue

            contract = Stock(symbol, 'SMART', 'USD')
            ib.qualifyContracts(contract)

            # Orden de apertura de posición corta: Se envía una orden de "SELL" (abre posición en corto)
            short_order = MarketOrder('SELL', qty)
            trade = ib.placeOrder(contract, short_order)
            print(f"Órdenes abiertas en IBKR: {ib.openOrders()}")
            print(f"Orden de short enviada para {symbol}, esperando ejecución...")

            order_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # Aquí podrías esperar la confirmación de llenado (similar a wait_for_order_fill) antes de colocar órdenes subordinadas

            # Definir los niveles para cubrir (take profit y stop loss) invertidos:
            entry_price = df['close'].iloc[-1]
            take_profit_price = round(entry_price * 0.95, 2)   # 5% por debajo del precio de entrada
            stop_loss_price = round(entry_price * 1.022, 2)      # 2.2% por encima del precio de entrada

            takeProfit = LimitOrder('BUY', qty, take_profit_price, tif='GTC', parentId=short_order.orderId, transmit=True)
            stopLoss = StopOrder('BUY', qty, stop_loss_price, tif='GTC', parentId=short_order.orderId, transmit=True)

            tp_trade = ib.placeOrder(contract, takeProfit)
            sl_trade = ib.placeOrder(contract, stopLoss)
            # ib.sleep(1)
            # assert tp_trade.orderStatus.status == 'Submitted'
            # assert tp_trade in ib.openTrades()
            # assert sl_trade.orderStatus.status == 'Submitted'
            # assert sl_trade in ib.openTrades()
            orders_df.loc[len(orders_df)] = [short_order.orderId, symbol, 'SHORT', qty, entry_price, 
                                               'Done', take_profit_price, stop_loss_price, order_date]
            print(f"Orden de TP (cubrimiento) enviada a {take_profit_price} y SL (cubrimiento) enviada a {stop_loss_price} para {symbol}.")

    await asyncio.gather(*tasks)
    await save_orders_to_excel()

# async def wait_for_order_fill(trade, contract, parent_order):
    # """ Espera hasta que la orden de compra se ejecute en IBKR. """
    # while trade.orderStatus.status not in ['Filled', 'Cancelled']:
    #     print(f"Esperando ejecución para {contract.symbol}...")
    #     await asyncio.sleep(2)

    # if trade.orderStatus.status == 'Filled':
    #     fill_price = trade.fills[0].price if trade.fills else None
    #     print(f"Orden ejecutada a {fill_price} para {contract.symbol}.")

    #     orders_df.loc[orders_df['OrderId'] == parent_order.orderId, 'Price'] = fill_price
    #     orders_df.loc[orders_df['OrderId'] == parent_order.orderId, 'Status'] = 'Filled'


async def save_orders_to_excel(): 
    """ Guarda las órdenes en un archivo Excel """
    orders_df.to_excel("orders_today.xlsx", index=False)

async def main_loop(): 
    await wait_for_market_open()
    while True:
        current_time = datetime.now().time()
        if current_time >= MARKET_CLOSE:
            print("El mercado ha cerrado. Finalizando operaciones.")
            break  #

        print("Actualizando datos...")
        await get_historical_data()
        combined_data = process_data()
        if not combined_data.empty:
            await execute_signals(SYMBOLS, combined_data) 
        print(f"Esperando {INTERVAL} segundos para la próxima actualización...")
        await asyncio.sleep(INTERVAL)
    await save_orders_to_excel()

if __name__ == "__main__":
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        print("Programa detenido.")

# # %%
