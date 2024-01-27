import concurrent
import math
import os
import datetime
import time
import decimal
import socket

import numpy as np
import pandas as pd
from binance import ThreadedWebsocketManager
from binance.client import Client

from configuration.my_logger import logger
from configuration.wrapper import timeit

API_KEY = os.environ.get("API_KEY")
SECRET_KEY = os.environ.get("SECRET_KEY")
MAIN_STABLE_COIN = "USDT"
symbol_live_update_dict = {}
UNWANTED_SYMBOLS = []


def symbol_live_update(new_data):
    ''' define how to process incoming WebSocket messages '''
    symbol = new_data['s']
    if new_data['e'] == 'error':
        logger.error(f"Theres been error collecting live data {symbol}")
        symbol_live_update_dict[symbol]['error'] = True
    else:
        symbol_live_update_dict[symbol]['price'] = new_data['b']
        symbol_live_update_dict[symbol]['error'] = False
        symbol_live_update_dict[symbol]['low'] = new_data['l']
        symbol_live_update_dict[symbol]['high'] = new_data['h']
        symbol_live_update_dict[symbol]['open'] = new_data['o']
        logger.info(f'{symbol} : {symbol_live_update_dict[symbol]}')


class Binance:
    # @timeit
    def __init__(self):
        try:
            self.client = Client(API_KEY, SECRET_KEY)
            self.min_quantity_per_coin = {}
            self.min_step_size_per_coin = {}
            self.all_symbols = []
            # self.get_all_coins()
            for symbol in self.all_symbols:
                self.get_min_qnt(symbol)
                self.get_min_step_size(symbol)
            self.bsm = None
            logger.info("Binance initialized")
        except Exception as e:
            logger.error(f"Cant initial binance object due to {e}")

    def get_bars_for_coin(self, coin, interval, historical_time, full_char_name=None, stable_coin=MAIN_STABLE_COIN):
        try:
            my_full_char_name = full_char_name
            if not my_full_char_name:
                my_full_char_name = f"{coin}{stable_coin}"
            bars = self.client.get_historical_klines(my_full_char_name, interval, historical_time, limit=1000)
            btc_df = pd.DataFrame(bars,
                                  columns=['start_date', 'open', 'high', 'low', 'close', 'volume', 'end_date', 'quote_asset_volume',
                                           'number_of_trades',
                                           'taker_buy_base', 'taker_buy_quote', 'ignore'])
            TO_NUM = ['open', 'high', 'low', 'close', 'volume', 'number_of_trades', 'taker_buy_base', 'taker_buy_quote', 'ignore']
            TO_DATE = ['start_date', 'end_date']
            for key in TO_NUM:
                btc_df[key] = pd.to_numeric(btc_df[key])
            for key in TO_DATE:
                btc_df[key] = pd.to_datetime(btc_df[key], unit='ms')
        except Exception as e:
            logger.error(f"Cant get bars for {coin} due to {e}")
            btc_df = pd.DataFrame()
        return btc_df

    def get_bars_for_coin_per_date(self, coin, interval, start_date, end_date, full_char_name=None, stable_coin=MAIN_STABLE_COIN):
        try:
            my_full_char_name = full_char_name
            if not my_full_char_name:
                my_full_char_name = f"{coin}{stable_coin}"
            bars = self.client.get_historical_klines(my_full_char_name, interval, str(start_date), str(end_date), limit=1000)
            btc_df = pd.DataFrame(bars,
                                  columns=['start_date', 'open', 'high', 'low', 'close', 'volume', 'end_date', 'quote_asset_volume',
                                           'number_of_trades',
                                           'taker_buy_base', 'taker_buy_quote', 'ignore'])
            TO_NUM = ['open', 'high', 'low', 'close', 'volume', 'number_of_trades', 'taker_buy_base', 'taker_buy_quote', 'ignore']
            TO_DATE = ['start_date', 'end_date']
            for key in TO_NUM:
                btc_df[key] = pd.to_numeric(btc_df[key])
            for key in TO_DATE:
                btc_df[key] = pd.to_datetime(btc_df[key], unit='ms')
        except Exception as e:
            logger.error(f"Cant get bars for {coin} due to {e}")
            btc_df = pd.DataFrame()
        # TODO: ADD real time candle
        return btc_df

    # @timeit
    def get_all_coins(self, stable_coin=MAIN_STABLE_COIN):
        if not self.all_symbols:
            try:
                futures_exchange_info = self.client.futures_exchange_info()
                trading_pairs = [info['baseAsset'] for info in futures_exchange_info['symbols'] if info['quoteAsset'] == stable_coin]
                self.all_symbols = trading_pairs
            except Exception as e:
                logger.error(f"Cant get all coins due to {e}")
                trading_pairs = []
        return self.all_symbols

    def get_min_qnt(self, symbol, stable_coin=MAIN_STABLE_COIN):
        try:
            if symbol not in self.min_quantity_per_coin.keys():
                info = self.client.get_symbol_info(f"{symbol}{stable_coin}")
                LOT_SIZE = info['filters'][2]
                self.min_quantity_per_coin[symbol] = float(LOT_SIZE['minQty'])
        except Exception as e:
            logger.error(f"Cant get min qnt due to {e}")
            return None
        return self.min_quantity_per_coin[symbol]

    def get_min_step_size(self, symbol, stable_coin=MAIN_STABLE_COIN):
        try:
            if symbol not in self.min_step_size_per_coin.keys():
                info = self.client.get_symbol_info(f"{symbol}{stable_coin}")
                LOT_SIZE = info['filters'][2]
                step_size = float(LOT_SIZE['stepSize'])
                self.min_step_size_per_coin[symbol] = int(round(-math.log(step_size, 10), 0))
        except Exception as e:
            logger.error(f"Cant get min step size due to {e}")
            return None
        return self.min_step_size_per_coin[symbol]

    # @timeit
    def get_open_orders(self):
        try:
            open_orders = self.client.futures_get_open_orders()
        except Exception as e:
            logger.error(f"Cant get open orders due to {e}")
            open_orders = []
        return open_orders

    def get_future_account_info(self):
        try:
            future_assets = self.client.futures_account_balance()
            future_asset_df = pd.DataFrame(future_assets)
            future_asset_df["updateTime"] = pd.to_datetime(future_asset_df["updateTime"], unit='ms')
            future_asset_df["balance"] = pd.to_numeric(future_asset_df["balance"])
        except Exception as e:
            logger.error(f"Cant get open orders due to {e}")
            future_asset_df = pd.DataFrame()
        return future_asset_df

    def get_future_positions_info(self):
        try:
            future_assets = self.client.futures_account()
            future_asset_df = pd.DataFrame(future_assets["positions"])
            future_asset_df["initialMargin"] = pd.to_numeric(future_asset_df["initialMargin"])
            future_asset_df["updateTime"] = pd.to_datetime(future_asset_df["updateTime"], unit='ms')
            future_asset_df = future_asset_df.loc[future_asset_df.initialMargin > 0]
        except Exception as e:
            logger.error(f"Cant get open orders due to {e}")
            future_asset_df = pd.DataFrame()
        return future_asset_df

    def get_future_account_trades(self):
        # TODO: NOT WORKING
        try:
            future_assets = self.client.futures_coin_historical_trades(symbol="BUSD")
        except Exception as e:
            logger.error(f"Cant get open orders due to {e}")
            future_asset_df = pd.DataFrame()
        return future_asset_df

    # @timeit
    def get_all_orders(self):
        try:
            all_orders = self.client.futures_get_all_orders()
        except Exception as e:
            logger.error(f"Cant get all orders due to {e}")
            all_orders = []
        return all_orders

    # @timeit
    def order_by_limit(self, symbol, side, quantity, price, demo=True, stable_coin=MAIN_STABLE_COIN):
        try:
            price = np.format_float_positional(price, trim='-')
            if demo:
                details = self.client.create_test_order(
                    symbol=f'{symbol}{stable_coin}',
                    side=side,
                    type='LIMIT',
                    timeInForce='GTC',
                    quantity=quantity,
                    price=price)
            else:
                details = self.client.futures_create_order(
                    symbol=f'{symbol}{stable_coin}',
                    side=side,
                    type='LIMIT',
                    timeInForce='GTC',
                    quantity=quantity,
                    price=price)
                logger.info(details)
        except Exception as e:
            logger.error(f"Cant order by limit due to {e}")
            details = {}
        return details

    # @timeit
    def order_by_market(self, symbol, side, quote_order, demo=True, stable_coin=MAIN_STABLE_COIN):
        try:
            if demo:
                details = self.client.create_test_order(
                    symbol=f'{symbol}{stable_coin}',
                    side=side,
                    type='MARKET',
                    quoteOrderQty=quote_order)
            else:
                details = self.client.futures_create_order(
                    symbol=f'{symbol}{stable_coin}',
                    side=side,
                    type='MARKET',
                    quoteOrderQty=quote_order)
                logger.info(details)
        except Exception as e:
            logger.error(f"Cant order by market due to {e}")
            details = {}
        return details

    # @timeit
    def get_order_details(self, symbol, order_id, stable_coin=MAIN_STABLE_COIN):
        try:
            num_of_try = 10
            for i in range(num_of_try):
                order_details = self.client.futures_get_order(symbol=f"{symbol}{MAIN_STABLE_COIN}", orderId=order_id)
                if order_details['status'] == "FILLED":
                    print(f"avg price : {order_details['avgPrice']}")
                    print(f"quantity : {order_details['origQty']}")
                    return order_details
                time.sleep(0.1)
            raise Exception("Num of try are passed")
        except Exception as e:
            logger.error(f"Cant order by limit due to {e}")
            order_details = {}
            return order_details

    # @timeit
    def get_symbol_ticker(self, symbol, stable_coin=MAIN_STABLE_COIN):
        try:
            symbol_price = self.client.get_symbol_ticker(symbol=f"{symbol}{stable_coin}")
        except Exception as e:
            logger.error(f"cant start symbol ticker due to {e}")
        return symbol_price

    def start_symbol_ticker(self):
        try:
            self.bsm = ThreadedWebsocketManager()
            self.bsm.start()
            logger.info("bsm started")
        except Exception as e:
            logger.error(f"cant start symbol ticker due to {e}")

    # @timeit
    def add_coin_to_symbol_ticker(self, coin, stable_coin=MAIN_STABLE_COIN):
        try:
            full_name = f'{coin}{stable_coin}'
            if full_name not in symbol_live_update_dict:
                symbol_live_update_dict[full_name] = {}
            self.bsm.start_symbol_ticker_socket(callback=symbol_live_update, symbol=full_name)
            logger.info(f"Symbol ticker for {coin} as started")
            return True
        except Exception as e:
            logger.error(f"Cant start symbol ticker to {coin} due to {e}")
            return False

    # @timeit
    def get_bar_for_multiple_coins(self, coin_list, interval, historical_time, full_char_name=None, stable_coin=MAIN_STABLE_COIN):
        try:
            main_list = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
                future_to_url = {executor.submit(self.get_bars_for_coin, coin, interval, historical_time,
                                                 full_char_name=full_char_name, stable_coin=stable_coin): coin for coin in coin_list}
                for future in concurrent.futures.as_completed(future_to_url):
                    coin = future_to_url[future]
                    try:
                        historical_df = future.result()
                        if not historical_df.empty:
                            main_list.append([coin, historical_df])
                    except Exception as exc:
                        logger.error(f'cant get data from {coin} due to {exc}')
                        continue
            full_coins_df = pd.DataFrame(main_list, columns=['coin', 'indicators'])
        except Exception as e:
            logger.error(f"cant get bar for multiple coins due to {e}")
            full_coins_df = pd.DataFrame()
        return full_coins_df

    def get_symbol_ticker_for_multiple_coins(self, coin_list, stable_coin=MAIN_STABLE_COIN):
        try:
            main_list = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
                future_to_url = {executor.submit(self.get_symbol_ticker, coin, stable_coin=stable_coin): coin for coin in coin_list}
                for future in concurrent.futures.as_completed(future_to_url):
                    coin = future_to_url[future]
                    try:
                        historical_df = future.result()
                        if historical_df:
                            main_list.append([coin, float(historical_df["price"])])
                    except Exception as exc:
                        logger.error(f'cant get data from {coin} due to {exc}')
                        continue
            full_coins_df = pd.DataFrame(main_list, columns=['coin', 'price'])
        except Exception as e:
            logger.error(f"cant get bar for multiple coins due to {e}")
            full_coins_df = pd.DataFrame()
        return full_coins_df

    def get_bar_for_multiple_coins_per_date(self, coin_list, interval, start_date, end_date, full_char_name=None, stable_coin=MAIN_STABLE_COIN):
        try:
            main_list = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
                future_to_url = {executor.submit(self.get_bars_for_coin_per_date, coin, interval, start_date, end_date,
                                                 full_char_name=full_char_name, stable_coin=stable_coin): coin for coin in coin_list}
                for future in concurrent.futures.as_completed(future_to_url):
                    coin = future_to_url[future]
                    try:
                        historical_df = future.result()
                        if not historical_df.empty:
                            main_list.append([coin, historical_df])
                    except Exception as exc:
                        logger.error(f'cant get data from {coin} due to {exc}')
                        continue
            full_coins_df = pd.DataFrame(main_list, columns=['coin', 'indicators'])
        except Exception as e:
            logger.error(f"cant get bar for multiple coins due to {e}")
            full_coins_df = pd.DataFrame()
        return full_coins_df

    # @timeit
    def get_bar_for_multiple_intervals(self, coin, intervals, historical_time, full_char_name=None, stable_coin=MAIN_STABLE_COIN):
        try:
            main_list = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
                future_to_url = {executor.submit(self.get_bars_for_coin, coin, interval, historical_time,
                                                 full_char_name=full_char_name, stable_coin=stable_coin): interval for interval in intervals}
                for future in concurrent.futures.as_completed(future_to_url):
                    interval = future_to_url[future]
                    try:
                        historical_df = future.result()
                        if not historical_df.empty:
                            main_list.append([interval, historical_df])
                    except Exception as exc:
                        print(f'cant get data from {coin} due to {exc}')
                        continue
            full_coins_df = pd.DataFrame(main_list, columns=['interval', 'indicators'])
        except Exception as e:
            logger.error(f"cant get bar for multiple intervals due to {e}")
            full_coins_df = pd.DataFrame()
        return full_coins_df

    # @timeit
    def get_bar_for_multiple_coin_multiple_intervals(self, coin_list, intervals, historical_time, full_char_name=None, stable_coin=MAIN_STABLE_COIN):
        try:
            main_list = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
                future_to_url = {executor.submit(self.get_bar_for_multiple_coins, coin_list, interval, historical_time,
                                                 full_char_name=full_char_name, stable_coin=stable_coin): interval for interval in intervals}
                for future in concurrent.futures.as_completed(future_to_url):
                    interval = future_to_url[future]
                    try:
                        historical_df = future.result()
                        if not historical_df.empty:
                            main_list.append([interval, historical_df])
                    except Exception as exc:
                        print(f'cant get data from {interval} due to {exc}')
                        continue
            full_coins_df = pd.DataFrame(main_list, columns=['interval', 'data'])
        except Exception as e:
            logger.error(f"cant get bar for multiple coin multiple intervals due to {e}")
            full_coins_df = pd.DataFrame()
        return full_coins_df

    def get_bar_for_multiple_coin_multiple_intervals_per_date(self, coin_list, intervals, start_date, end_date, full_char_name=None,
                                                              stable_coin=MAIN_STABLE_COIN):
        try:
            main_list = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
                future_to_url = {executor.submit(self.get_bar_for_multiple_coins_per_date, coin_list, interval, start_date, end_date,
                                                 full_char_name=full_char_name, stable_coin=stable_coin): interval for interval in intervals}
                for future in concurrent.futures.as_completed(future_to_url):
                    interval = future_to_url[future]
                    try:
                        historical_df = future.result()
                        if not historical_df.empty:
                            main_list.append([interval, historical_df])
                    except Exception as exc:
                        print(f'cant get data from {interval} due to {exc}')
                        continue
            full_coins_df = pd.DataFrame(main_list, columns=['interval', 'data'])
        except Exception as e:
            logger.error(f"cant get bar for multiple coin multiple intervals due to {e}")
            full_coins_df = pd.DataFrame()
        return full_coins_df


BINANCE_API = Binance()


def get_future_balance_update(stable_coin):
    future_balance_df = BINANCE_API.get_future_account_info()
    future_balance_df = future_balance_df.drop(columns=['accountAlias'])

    balance_only_stable_coins = future_balance_df.loc[future_balance_df.asset.isin(['USDT', 'BUSD', 'USDC', 'USDP', 'BUSD', 'TUSD'])]
    total_worth = balance_only_stable_coins.balance.sum()

    balance_without_stable_coins = future_balance_df.loc[~future_balance_df.asset.isin(['USDT', 'BUSD', 'USDC', 'USDP', 'BUSD', 'TUSD'])]
    symbol_to_price_df = BINANCE_API.get_symbol_ticker_for_multiple_coins(list(balance_without_stable_coins.asset), stable_coin)
    balance_without_stable_coins = balance_without_stable_coins.merge(symbol_to_price_df, left_on='asset', right_on='coin')
    balance_without_stable_coins["current_worth"] = np.multiply(balance_without_stable_coins.balance, balance_without_stable_coins.price)

    total_worth += balance_without_stable_coins.current_worth.sum()
    only_active_asset = future_balance_df.loc[future_balance_df.balance > 0]
    return total_worth, only_active_asset


def get_future_open_positions():
    future_trade_df = BINANCE_API.get_future_positions_info()
    trade_df_to_return = future_trade_df[['symbol', 'initialMargin', 'entryPrice', 'leverage', 'updateTime']].copy()

    return trade_df_to_return


if __name__ == '__main__':
    get_future_open_positions()
