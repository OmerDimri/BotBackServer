import datetime
import warnings

import numpy as np
from src.api.binance_api import BINANCE_API
from configuration.my_logger import logger

warnings.filterwarnings(action='ignore', category=FutureWarning)


def convert_interval_to_datetime(interval):
    value = int(interval[:-1])
    interval_name = interval[-1]
    if interval_name == "m":
        return datetime.timedelta(minutes=value)
    if interval_name == "h":
        return datetime.timedelta(hours=value)
    if interval_name == "d":
        return datetime.timedelta(days=value)
    if interval_name == "W":
        return datetime.timedelta(weeks=value)
    if interval_name == "M":
        return datetime.timedelta(days=value * 31)
    logger.error(f"There arent interval here {interval}")


def get_bars_per_position_live(coin, stable, interval, bar_amount, start_date):
    start_date = datetime.datetime.fromtimestamp(start_date)
    end_date = datetime.datetime.now()

    date_start_from = start_date - convert_interval_to_datetime(interval) * bar_amount
    date_end_to = end_date + convert_interval_to_datetime(interval) * bar_amount
    if date_end_to > datetime.datetime.now():
        date_end_to = datetime.datetime.now()

    bars = BINANCE_API.get_bars_for_coin_per_date(coin, interval, date_start_from, date_end_to, stable_coin=stable)
    return bars[["start_date", "open", "high", "low", "close"]]


def get_bars_per_position_done(coin, stable, interval, bar_amount, start_date, end_date):
    date_start_from = start_date - convert_interval_to_datetime(interval) * bar_amount
    date_end_to = end_date + convert_interval_to_datetime(interval) * bar_amount
    if date_end_to > datetime.datetime.now():
        date_end_to = datetime.datetime.now()

    bars = BINANCE_API.get_bars_for_coin_per_date(coin, interval, date_start_from, date_end_to, stable_coin=stable)
    return bars[["start_date", "open", "high", "low", "close"]]


def get_relevant_price_list(symbol_list, stable="USDT"):
    return BINANCE_API.get_symbol_ticker_for_multiple_coins(symbol_list, stable)


def add_transaction_df_to_live(transaction_df, symbol_to_price_df, coin):
    coin_balance_hand = 0
    for single_transaction in transaction_df.itertuples():
        if single_transaction.type == "BUY":
            coin_balance_hand += float(single_transaction.amount)
        else:
            coin_balance_hand -= float(single_transaction.amount)
    new_transaction_row = transaction_df.iloc[0].copy()
    new_transaction_row.amount = np.absolute(coin_balance_hand)
    new_transaction_row.price = symbol_to_price_df.loc[symbol_to_price_df.coin == coin].price.iloc[0]
    if coin_balance_hand < 0:
        new_transaction_row.type = "BUY"
    else:
        new_transaction_row.type = "SELL"
    transaction_df = transaction_df.append([new_transaction_row], ignore_index=True)
    return transaction_df
