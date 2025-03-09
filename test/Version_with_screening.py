# ######
#Run screening
with open("screening_tendencias_final.py") as f:
    exec(f.read())
print("Screening completo")
del df_data
del df_data1
del df
########


import asyncio
import pandas as pd
import numpy as np
import ta
import warnings
from datetime import datetime, time
from pytz import timezone
from ib_insync import IB, Stock, LimitOrder, StopOrder, util

warnings.filterwarnings("ignore")
util.startLoop()

CONFIG = {
    'SYMBOLS_CSV': 'symbols.csv',
    'IB_HOST': '127.0.0.1',
    'IB_PORT': 4002, 
    'CLIENT_ID': 31322,
    'INTERVAL': 60, 
    'DURATION': '30 D', 
    'BAR_SIZE': '30 mins',
    'MAX_CONCURRENT_REQUESTS': 5, ###### se puede amplificar a mas?
    'MARKET_OPEN': time(9, 30, tzinfo=timezone('US/Eastern')),
    'MARKET_CLOSE': time(16, 0, tzinfo=timezone('US/Eastern')),
    'SLIPPAGE': 0.05,
    'TAKE_PROFIT_RATIO': 0.04,
    'STOP_LOSS_RATIO': 0.02,
    'MAX_POSITION_SIZE': 1000, #### chequear cuando nos den el go
    'USE_REALTIME_DATA': True 
}

class TradingState:
    def __init__(self):
        self.orders = pd.DataFrame(columns=[
            'OrderId', 'Symbol', 'Action', 'Quantity', 'Entry',
            'TakeProfit', 'StopLoss', 'RiskReward', 'Status', 'Date'
        ])
        self.historical_data = {}
        self.real_time_prices = {}  
        self.symbols = []
        
trading_state = TradingState()
ib = IB()

def load_symbols():
    try:
        df = pd.read_csv(CONFIG['SYMBOLS_CSV'])
        valid_symbols = []
        
        for symbol in df['Symbol'].unique():
            try:
                contract = Stock(symbol, 'SMART', 'USD')
                ib.qualifyContracts(contract)
                valid_symbols.append(symbol)
                print(f"Valid symbol: {symbol}")
            except Exception as e:
                print(f"Invalid symbol: {symbol} - {str(e)}") ### plantear  retirar en  el futuro OPtimizar

        trading_state.symbols = valid_symbols
        print(f"Active symbols: {valid_symbols}")
        
    except Exception as e:
        print(f"Error loading CSV: {e}")
        trading_state.symbols = []

def connect_ib():
    try:
        ib.connect(
            CONFIG['IB_HOST'],
            CONFIG['IB_PORT'],
            clientId=CONFIG['CLIENT_ID']
        )
        print("\n" + "="*50)
        print("Connected to Interactive Brokers")
        load_symbols()
    except Exception as e:
        print(f"Connection error: {e}")
        raise

async def fetch_historical_data(symbol):
    try:
        contract = Stock(symbol, 'SMART', 'USD')
        ib.qualifyContracts(contract)
        
        bars = await ib.reqHistoricalDataAsync(
            contract=contract,
            endDateTime="",
            durationStr=CONFIG['DURATION'],
            barSizeSetting=CONFIG['BAR_SIZE'],
            whatToShow='MIDPOINT',
            useRTH=False,
            timeout=60
        )
        
        if bars:
            df = util.df(bars)
            df['date'] = pd.to_datetime(df['date'])
            return symbol, df
            
        return symbol, None 
        
    except Exception as e:
        print(f"Error getting data for {symbol}: {e}") 
        return symbol, None
    
    
async def get_historical_data():
    semaphore = asyncio.Semaphore(CONFIG['MAX_CONCURRENT_REQUESTS'])
    
    async def limited_fetch(symbol):
        async with semaphore:
            return await fetch_historical_data(symbol)
    
    tasks = [limited_fetch(symbol) for symbol in trading_state.symbols]
    results = await asyncio.gather(*tasks)
    
    for symbol, data in results:
        if data is not None:
            trading_state.historical_data[symbol] = data
            print(f"Data for {symbol} ({len(data)} records):")
            print(f"First record: {data['date'].iloc[0]}")
            print(f"Last record: {data['date'].iloc[-1]}")
            print(f"Last close: {data['close'].iloc[-1]:.2f}")

async def update_real_time_prices(symbol):
    """Actualiza los precios en tiempo real para un símbolo"""
    try:
        contract = Stock(symbol, 'SMART', 'USD')
        ticker = await ib.reqMktDataAsync(contract)
        
        trading_state.real_time_prices[symbol] = {
            'open': ticker.open if not np.isnan(ticker.open) else None,
            'close': ticker.close if not np.isnan(ticker.close) else None,
            'marketPrice': ticker.marketPrice if not np.isnan(ticker.marketPrice) else None
        }
        
    except Exception as e:
        print(f"Error updating real-time prices for {symbol}: {e}")

def calculate_indicators(df): 
  
    # Cálculo de indicadores técnicos
    df['RSI'] = ta.momentum.rsi(close=df['close'], window=20)
    df['SMA100'] = df['close'].rolling(window=100, min_periods=100).mean()

    # RSI en sobreventa
    df['RSI_oversold_last5'] = (df['RSI'].rolling(window=20, min_periods=20).min() < 35).astype(int)
    # Rango máximo-mínimo
    df['max_min_range'] = df['high'].rolling(window=20, min_periods=20).max() - df['low'].rolling(window=20, min_periods=20).min()
    # Pendiente positiva de la SMA100 (mirando hacia atrás)
    df['SMA100_slope'] = ((df['SMA100'] - df['SMA100'].shift(1)) >= 0).rolling(window=5, min_periods=5).sum() == 5
    # Señal de ruptura de SMA100
    df['break_SMA'] = (
            (df['close'] > df['SMA100']) &
            # (df['close'].rolling(window=20, min_periods=20).std() <= df['close'] * 0.05) &
            # (df['SMA100_slope']) &
            (df['max_min_range'] <= df['max_min_range'].quantile(0.5)) &
            (df['close'].shift(1) <= df['SMA100'].shift(1)) &
            (df['close'].shift(2) <= df['SMA100'].shift(2)) &
            (df['close'].shift(3) <= df['SMA100'].shift(3)) &
            (df['close'].shift(4) <= df['SMA100'].shift(4))).astype(int)

    df['buy_signal'] = np.where(((df['RSI_oversold_last5'] == 1) & (df['break_SMA'] == 1) ), 
                True, False)    
    return df

async def place_order(symbol, action):
    try:
        contract = Stock(symbol, 'SMART', 'USD')
        await ib.qualifyContractsAsync(contract)
        
        # Obtener precios actualizados
        rt_prices = trading_state.real_time_prices.get(symbol, {})
        current_price = rt_prices.get('marketPrice')
        
        if current_price is None or np.isnan(current_price):
            print(f"No valid price data for {symbol}")
            return False
        
        # Cálculo de tamaño de posición ###### SIZING FIJO X TRADE IMPLEMENTAR
        # account = await ib.accountSummaryAsync()
        # equity = next((a.value for a in account if a.tag == 'NetLiquidation'), 0) #En prueva (Size dinamico segun el dinero de la cuenta)
        # risk_amount = float(equity) * CONFIG['RISK_PER_TRADE']
        #price_diff = CONFIG['STOP_LOSS_RATIO'] * current_price
        #position_size = min(int(risk_amount / price_diff), CONFIG['MAX_POSITION_SIZE'] // current_price) ##### DEFINIR DE MANERA SIMPLE COMO LA IPLEMENTACIÓN EN DEMO
        position_size = CONFIG['MAX_POSITION_SIZE'] // current_price #division entera - sizing fijo. OJO: MAX_POSITION_SIZE = 1000
        
        if position_size < 1:
            print(f"Position too small for {symbol}. Skipping.")
            return False

        # Cálculo de TP/SL
        take_profit = round(current_price * (1 + CONFIG['TAKE_PROFIT_RATIO'] 
                        if action == 'BUY' else 1 - CONFIG['TAKE_PROFIT_RATIO']), 2) # DEFINIR QUE SOLO SEAN VENTAS EN LARGO
        
        stop_loss = round(current_price * (1 - CONFIG['STOP_LOSS_RATIO'] 
                      if action == 'BUY' else 1 + CONFIG['STOP_LOSS_RATIO']), 2) ######### EST ABIEN DEFINIDO SOLO QUE TENENMOS QUE PONER Y TENER SETEADO LOS PORCENTAJES EJM: PARA 1k EL REIESGO SERIA DE 200 DOLARES

        # Crear orden bracket
        parent = LimitOrder(action, position_size, current_price)
        parent.transmit = False
        
        takeProfit = LimitOrder('SELL' if action == 'BUY' else 'BUY', 
                               position_size, take_profit)
        takeProfit.transmit = False
        
        stopLoss = StopOrder('SELL' if action == 'BUY' else 'BUY', 
                           position_size, stop_loss)
        stopLoss.transmit = True
        
        # Enviar órdenes
        await ib.placeOrderAsync(contract, parent)
        await ib.placeOrderAsync(contract, takeProfit)
        await ib.placeOrderAsync(contract, stopLoss)

        # Registrar orden
        new_order = {
            'OrderId': parent.orderId,
            'Symbol': symbol,
            'Action': action,
            'Quantity': position_size,
            'Entry': current_price,
            'TakeProfit': take_profit,
            'StopLoss': stop_loss,
            'RiskReward': f"{1}:{round(CONFIG['TAKE_PROFIT_RATIO']/CONFIG['STOP_LOSS_RATIO'])}", ########## ACTUALIAR EL REGISTRO CON LAS CORREXCIONES
            'Status': 'Submitted',
            'Date': datetime.now(timezone('US/Eastern')).strftime("%Y-%m-%d %H:%M:%S")
        }
        
        trading_state.orders = pd.concat([
            trading_state.orders,
            pd.DataFrame([new_order])
        ], ignore_index=True) ######################### QUE ESESTO REGISTRO??¡ ES NECESARIO?, LA VERFICACIÓN DE LA EJECUSION SERA MANUAL JUNTO CON EL MONITOREO

        print(f"Bracket order placed for {symbol}: {new_order}")
        return True
        
    except Exception as e:
        print(f"Order error for {symbol}: {e}")
        return False

async def process_symbol(symbol): ################## ES ESTA PARTE QUE TAN NECESARIA ES? LE PORCESO DE VALIDA CION SRIA MANUAL
    try:
        df = trading_state.historical_data.get(symbol)
        if df is None or len(df) < 100:
            print(f"Insufficient data for {symbol}")
            return
            
        df = calculate_indicators(df.copy())
        
        print(f"Analysis for {symbol}:")
        print(f"Last close: {df['close'].iloc[-1]:.2f}")
        print(f"SMA100: {df['SMA100'].iloc[-1]:.2f}")
        print(f"RSI: {df['RSI'].iloc[-1]:.2f}")
        print(f"Break SMA: {df['break_SMA'].iloc[-1]}")
        print(f"RSI Oversold: {df['RSI_oversold'].iloc[-1]}")
        print(f"Buy signal: {df['buy_signal'].iloc[-1]}")
        
        if df['buy_signal'].iloc[-1] and CONFIG['USE_REALTIME_DATA']:
            # Verificación en tiempo real
            rt_data = trading_state.real_time_prices.get(symbol, {})
            market_price = rt_data.get('marketPrice')
            last_close = df['close'].iloc[-1]
            
            if market_price and not np.isnan(market_price) and market_price > last_close:
                print(f"Valid buy signal for {symbol} at {market_price}")
                await place_order(symbol, 'BUY')
            else:
                print(f"Price validation failed for {symbol}")
                
        elif df['buy_signal'].iloc[-1]:
            # Modo sin verificación en tiempo real
            await place_order(symbol, 'BUY')
            
    except Exception as e:
        print(f"Processing error for {symbol}: {e}")

async def trading_cycle():
    print("\n" + "="*50)
    print(f"Starting cycle: {datetime.now(timezone('US/Eastern')).strftime('%Y-%m-%d %H:%M:%S')}")
    
    await get_historical_data()
    
    if CONFIG['USE_REALTIME_DATA']:
        print("\nUpdating real-time prices...")
        await asyncio.gather(*[update_real_time_prices(s) for s in trading_state.symbols])
    
    print("\nAnalyzing signals...")
    tasks = [process_symbol(symbol) for symbol in trading_state.symbols]
    await asyncio.gather(*tasks)
    
    print("\nOrder summary:")
    print(trading_state.orders.tail())

async def market_monitoring():
    est = timezone('US/Eastern')
    print("\nWaiting for market open...")
    
    while True:
        now = datetime.now(est)
        current_time = now.time()
        
        if current_time >= CONFIG['MARKET_CLOSE']:
            print("\nMarket closed. Ending operations.")
            break
            
        if CONFIG['MARKET_OPEN'] <= current_time < CONFIG['MARKET_CLOSE']:
            await trading_cycle()
            print(f"\nNext cycle in {CONFIG['INTERVAL']//60} minutes...")
            await asyncio.sleep(CONFIG['INTERVAL'])
        else:
            await asyncio.sleep(60)

async def main():
    connect_ib()
    try:
        await market_monitoring()
    finally:
        ib.disconnect()
        print("\nDisconnected from Interactive Brokers")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram stopped by user")
        ib.disconnect()
