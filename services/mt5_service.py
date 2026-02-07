import logging
import platform

logger = logging.getLogger(__name__)

class MT5Service:
    def __init__(self):
        self.connected = False
        self.platform = platform.system()

    def connect(self, login=None, password=None, server=None):
        """Connect to MetaTrader 5 terminal."""
        if self.platform != "Windows":
            logger.warning(f"MetaTrader5 official Python library is only supported on Windows. Current platform: {self.platform}")
            return False

        try:
            import MetaTrader5 as mt5
            if not mt5.initialize(login=login, password=password, server=server):
                logger.error(f"MT5 initialization failed, error code: {mt5.last_error()}")
                return False
            self.connected = True
            logger.info("Connected to MetaTrader 5")
            return True
        except ImportError:
            logger.error("MetaTrader5 library not installed. Run 'pip install MetaTrader5'")
            return False
        except Exception as e:
            logger.error(f"Unexpected error connecting to MT5: {e}")
            return False

    def get_account_info(self):
        """Fetch account balance, equity, and margin."""
        if not self.connected:
            return {"error": "Not connected to MT5"}

        import MetaTrader5 as mt5
        acc = mt5.account_info()
        if acc is None:
            return {"error": "Failed to get account info"}

        return {
            "balance": acc.balance,
            "equity": acc.equity,
            "margin": acc.margin,
            "free_margin": acc.margin_free,
            "leverage": acc.leverage,
            "currency": acc.currency
        }

    def get_positions(self):
        """Get all open positions."""
        if not self.connected:
            return []

        import MetaTrader5 as mt5
        positions = mt5.positions_get()
        if positions is None:
            return []

        return [p._asdict() for p in positions]
    
    def get_symbol_data(self, symbol, timeframe, count=100):
        """Get historical data for a symbol."""
        if not self.connected:
            return None
            
        try:
            import MetaTrader5 as mt5
            import pandas as pd
            
            # Convert timeframe string to MT5 constants
            tf_map = {
                "1m": mt5.TIMEFRAME_M1,
                "5m": mt5.TIMEFRAME_M5,
                "15m": mt5.TIMEFRAME_M15,
                "30m": mt5.TIMEFRAME_M30,
                "1h": mt5.TIMEFRAME_H1,
                "4h": mt5.TIMEFRAME_H4,
                "1d": mt5.TIMEFRAME_D1
            }
            
            mt5_timeframe = tf_map.get(timeframe, mt5.TIMEFRAME_H1)
            
            # Request historical data
            rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, count)
            
            if rates is None or len(rates) == 0:
                logger.warning(f"No MT5 data received for {symbol}")
                return None
                
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High', 
                'low': 'Low',
                'close': 'Close',
                'tick_volume': 'Volume'
            })
            
            # Reorder columns
            df = df[['time', 'Open', 'High', 'Low', 'Close', 'Volume']]
            
            logger.info(f"Retrieved {len(df)} candles for {symbol} from MT5")
            return df.to_dict('records')
            
        except Exception as e:
            logger.error(f"Error getting MT5 symbol data: {e}")
            return None
    
    def get_symbol_info(self, symbol):
        """Get symbol information from MT5."""
        if not self.connected:
            return None
            
        try:
            import MetaTrader5 as mt5
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return None
                
            return {
                'symbol': symbol_info.name,
                'bid': symbol_info.bid,
                'ask': symbol_info.ask,
                'spread': symbol_info.spread,
                'point': symbol_info.point,
                'digits': symbol_info.digits,
                'volume_min': symbol_info.volume_min,
                'volume_max': symbol_info.volume_max,
                'volume_step': symbol_info.volume_step,
                'trade_mode': symbol_info.trade_mode
            }
        except Exception as e:
            logger.error(f"Error getting symbol info: {e}")
            return None
    
    def send_order(self, symbol, order_type, volume, price=None, sl=None, tp=None, comment=""):
        """Send a trading order to MT5."""
        if not self.connected:
            return {"error": "Not connected to MT5"}
            
        try:
            import MetaTrader5 as mt5
            
            # Prepare order request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": mt5.ORDER_TYPE_BUY if order_type.upper() == "BUY" else mt5.ORDER_TYPE_SELL,
                "price": price,
                "sl": sl,
                "tp": tp,
                "deviation": 20,
                "magic": 234000,
                "comment": comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Send the order
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return {
                    "error": f"Order failed with code {result.retcode}",
                    "comment": result.comment
                }
                
            return {
                "success": True,
                "order": result.order,
                "price": result.price,
                "volume": result.volume
            }
            
        except Exception as e:
            return {"error": f"Order error: {str(e)}"}
    
    def get_live_price(self, symbol):
        """Get current bid/ask prices for symbol."""
        if not self.connected:
            return None
            
        try:
            import MetaTrader5 as mt5
            tick = mt5.symbol_info_tick(symbol)
            
            if tick is None:
                return None
                
            return {
                'bid': tick.bid,
                'ask': tick.ask,
                'spread': tick.ask - tick.bid,
                'time': pd.to_datetime(tick.time, unit='s'),
                'last': tick.last if hasattr(tick, 'last') else tick.bid
            }
        except Exception as e:
            logger.error(f"Error getting live price: {e}")
            return None
