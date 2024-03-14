import datetime
import json
import time

import numpy as np
import pandas as pd
from sqlalchemy import select, or_, and_

from configuration.my_logger import logger
from src.api.db_api import get_table, convert_date_columns_to_str
from src.constant import FULL_KEYS_DICT, custom_serializer
from src.utils.generic_utils import convert_interval_to_datetime
from src.utils.mode_singelton import ModeSingleton
from src.utils.statistics_functions import analyze_basic_statistics, line_function, bar_function
from cachetools import TTLCache
import swifter

strategy_name_to_id = dict()
mode_object = ModeSingleton()
cache = TTLCache(maxsize=10, ttl=3600)


def analyze_transaction_by_type(transaction_df, c_type):
    type_df = transaction_df.loc[transaction_df.type == c_type].copy()
    type_df["percent_from_all"] = np.abs(type_df.amount_stable / type_df.amount_stable.sum())
    type_df["part_to_avg"] = np.multiply(type_df["percent_from_all"], type_df.price)
    return type_df


def adding_analyze_to_positions(row, transaction_of_row):
    transaction_of_row = transaction_of_row.loc[transaction_of_row.position_id == row.id].copy()
    transaction_of_row['amount_stable'] = np.where(transaction_of_row.type == "SELL",
                                                   np.multiply(transaction_of_row.amount, transaction_of_row.price),
                                                   -1 * np.multiply(transaction_of_row.amount, transaction_of_row.price))

    profit_before = transaction_of_row.amount_stable.sum()
    position_side = row.side

    sell_transactions = analyze_transaction_by_type(transaction_of_row, "SELL")
    buy_transactions = analyze_transaction_by_type(transaction_of_row, "BUY")
    current_type_transaction = buy_transactions
    if position_side == "SHORT":
        current_type_transaction = buy_transactions
    row['profit'] = transaction_of_row.amount_stable.sum() - transaction_of_row.amount_fee.sum()
    row['duration'] = row.end_date - row.start_date
    amount_stable = current_type_transaction.amount_stable.sum()
    if amount_stable:
        row['change_percent'] = np.divide(profit_before, np.abs(current_type_transaction.amount_stable.sum()))
    else:
        row['change_percent'] = 0
    row['total_invest'] = np.absolute(np.abs(current_type_transaction.amount_stable.sum()))
    row['avg_buy'] = buy_transactions.part_to_avg.sum()
    row['avg_sell'] = sell_transactions.part_to_avg.sum()
    row['fee_stable'] = transaction_of_row.amount_fee.sum()

    if row.id not in strategy_name_to_id:
        find_strategy_ids([], [], [row.strategy_id])
    row['strategy_name'] = strategy_name_to_id.get(row.strategy_id)
    return row


def find_strategy_ids(platform_ids, strategy_names, strategy_ids):
    engine_s, STRATEGY_TABLE = get_table("strategy")
    if strategy_ids or strategy_names:
        filter_condition = or_(
            STRATEGY_TABLE.columns.name.in_(strategy_names),
            STRATEGY_TABLE.columns.id.in_(strategy_ids)
        )
        if platform_ids:
            filter_condition = and_(
                filter_condition,
                STRATEGY_TABLE.columns.platform_id.in_(platform_ids)
            )
    else:
        if platform_ids:
            filter_condition = STRATEGY_TABLE.columns.platform_id.in_(platform_ids)
        else:
            return []
    strategy_query = STRATEGY_TABLE.select().where(filter_condition)
    results = engine_s.execute(strategy_query).fetchall()
    final_results = []
    for result in results:
        strategy_id = result[0]
        strategy_name = result[2]
        if strategy_id not in strategy_name_to_id:
            strategy_name_to_id[strategy_id] = strategy_name
        final_results.append(str(strategy_id))
    return final_results


def get_position_records_from_db(data_dict):
    engine, POSITION_TABLE = get_table("position")
    # DATE CONDITION
    if "start_date" not in data_dict:
        return False
    start_date = data_dict.get("start_date")
    start_date = datetime.datetime.strptime(start_date, "%Y-%m-%dT%H:%M:%S.%fZ")
    if "end_date" not in data_dict:
        end_date = datetime.datetime.now()
    else:
        end_date = data_dict.get("end_date")
        end_date = datetime.datetime.strptime(end_date, "%Y-%m-%dT%H:%M:%S.%fZ")
    date_by = data_dict.get("date_by", "start_date")
    if date_by == "start_date":
        filter_condition = POSITION_TABLE.columns.start_date.between(start_date, end_date)
    else:
        filter_condition = POSITION_TABLE.columns.end_date.between(start_date, end_date)

    # Strategy & platform Filters
    platform_ids = data_dict.get("platform_ids", [])
    strategy_ids = data_dict.get("strategy_ids", [])
    strategy_names = data_dict.get("strategy_names", [])
    if platform_ids or strategy_ids or strategy_names:
        strategy_ids_to_filter_by = find_strategy_ids(platform_ids, strategy_names, strategy_ids)
        filter_condition = and_(
            filter_condition,
            POSITION_TABLE.columns.strategy_id.in_(strategy_ids_to_filter_by)
        )
    # Coins
    coins = data_dict.get("coins", [])
    if coins:
        filter_condition = and_(
            filter_condition,
            POSITION_TABLE.columns.symbol.in_(coins)
        )
    # End_By
    end_by = data_dict.get("end_by", [])
    if end_by:
        if isinstance(end_by, str):
            end_by = [end_by]
        filter_condition = and_(
            filter_condition,
            POSITION_TABLE.columns.end_by.in_(end_by)
        )
    # SIDE
    side = data_dict.get("side", [])
    if side.lower() == "both":
        side = []
    elif isinstance(side, str):
        side = [side.upper()]
    if side:
        filter_condition = and_(
            filter_condition,
            POSITION_TABLE.columns.side.in_(side)
        )

    filter_condition = and_(
        filter_condition,
        POSITION_TABLE.columns.end_by != ''
    )

    strategy_query = POSITION_TABLE.select().filter(filter_condition)
    result = engine.execute(strategy_query).fetchall()
    result = pd.DataFrame(result, columns=FULL_KEYS_DICT["position"])

    dates_for_csv = data_dict.get("dates_from_csv", [])
    if dates_for_csv:
        dates_for_csv = [(datetime.datetime.fromisoformat(d[0]), datetime.datetime.fromisoformat(d[1])) for d in dates_for_csv]
        result = result.loc[[any((row.start_date >= start and row.start_date <= end) for start, end in dates_for_csv) for _, row in result.iterrows()]]
    print(result)

    if result.empty:
        return result
    return result


def get_transaction_by_position_ids(position_ids):
    engine, TRANSACTION_TABLE = get_table("transaction")
    filter_condition = TRANSACTION_TABLE.columns.position_id.in_(position_ids)
    strategy_query = TRANSACTION_TABLE.select().filter(filter_condition)
    result = engine.execute(strategy_query).fetchall()
    result = pd.DataFrame(result, columns=FULL_KEYS_DICT["transaction"])
    return result


def add_analyzing_to_positions(position_df):
    position_ids = list(position_df.id)
    logger.info(f"got {position_df.shape[0]} positions!")
    logger.info("got transaction from DB")
    transaction_df = get_transaction_by_position_ids(position_ids)

    start_time = time.time()
    position_df = position_df.swifter.apply(adding_analyze_to_positions, axis=1, args=[transaction_df])
    position_df = position_df.sort_values(by=['id'], ascending=False)
    end_time = time.time()
    avg_per_pos = np.divide(end_time - start_time, position_df.shape[0])
    logger.info(f"Add Analyze to each position, avg per pos {avg_per_pos}")
    return position_df


def get_position_and_basic_analysis(query_request, refresh):
    query_key = json.dumps(query_request, sort_keys=True)
    if refresh or query_key not in cache:
        position_df = get_position_records_from_db(query_request)
        logger.info("Got Pos from DB")
        position_df = add_analyzing_to_positions(position_df)
        logger.info("Got Analyze for Pos")
        position_df = filter_position_by_indicator_from_query(position_df, query_request)
        basic_statistics = analyze_basic_statistics(position_df)
        final_answer = {
            "positions": position_df,
            "basic_statistics": basic_statistics
        }
        cache[query_key] = final_answer
    return cache[query_key]


def filter_position_by_indicator_from_query(position_df, query_request):
    if query_request.get("max_profit") or query_request.get("min_profit"):
        min_profit, max_profit = query_request.get("min_profit"), query_request.get("max_profit")
        if min_profit:
            min_profit = float(min_profit)
            position_df = position_df.loc[position_df.profit > min_profit]
        if max_profit:
            max_profit = float(max_profit)
            position_df = position_df.loc[position_df.profit < max_profit]
    if query_request.get("min_duration") or query_request.get("max_duration"):
        min_duration, max_duration = query_request.get("min_duration"), query_request.get("max_duration")
        if min_duration:
            min_duration = convert_interval_to_datetime(min_duration)
            position_df = position_df.loc[position_df.duration > min_duration]
        if max_duration:
            max_duration = convert_interval_to_datetime(max_duration)
            position_df = position_df.loc[position_df.duration < max_duration]
    return position_df


graph_function_to_type = {
    "line": line_function,
    "bar": bar_function
}

if __name__ == '__main__':
    # GET POSITION &
    my_request = {
        "query": {
            "start_date": '2020-01-22T00:00:00.653Z',
            "platform_ids": [131]
        },
        "refresh": False,
        "graph_type": "line",
        "graph_interval": "1h",
        "graph_by": "symbol",
        "mode": "staging"
    }
    mode_object.set_mode(my_request["mode"])
    answer_as_dict = get_position_and_basic_analysis(my_request["query"], my_request["refresh"])

    graph_interval = my_request.get("graph_interval")
    graph_type = my_request.get("graph_type")
    graph_by = my_request.get("graph_by")
    if graph_by and graph_type and graph_interval:
        answer_as_dict["graph_data"] = graph_function_to_type[graph_type](answer_as_dict["positions"], graph_interval, graph_by)

    for key, value in answer_as_dict.items():
        if isinstance(value, pd.DataFrame):
            answer_as_dict[key] = value.to_dict("records")

    final_answer = json.dumps(answer_as_dict, default=custom_serializer)
    print(1)
