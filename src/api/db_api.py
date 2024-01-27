import concurrent
import datetime
import os
from operator import or_

import pandas as pd
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, FLOAT, and_, Date, DateTime, update, delete
from sqlalchemy import insert, select
import numpy as np
from sqlalchemy.testing import in_, eq_regex

from src.api.binance_api import Binance
from src.constant import FULL_KEYS_DICT
from src.utils.generic_utils import get_relevant_price_list, add_transaction_df_to_live
from src.utils.mode_singelton import ModeSingleton

DB_IP = os.environ["DB_IP"]
engine_url_dev = f'postgresql+psycopg2://postgres:123456@{DB_IP}/bot_dev'
engine_dev = create_engine(engine_url_dev)
engine_dev.connect()
metadata_dev = MetaData(engine_dev)

engine_url_stg = f'postgresql+psycopg2://postgres:123456@{DB_IP}/bot_stg'
engine_stg = create_engine(engine_url_stg)
engine_stg.connect()
metadata_stg = MetaData(engine_stg)
mode_object = ModeSingleton()

engine_url_prd = f'postgresql+psycopg2://postgres:123456@{DB_IP}/bot_prd'
engine_prd = create_engine(engine_url_prd)
engine_prd.connect()
metadata_prd = MetaData(engine_prd)
mode_object = ModeSingleton()


def get_table(table_name):
    mode = mode_object.get_mode()
    if mode == "development":
        engine = engine_dev
    elif mode == "production":
        engine = engine_prd
    else:
        engine = engine_stg
    engine.connect()
    metadata = MetaData(engine)
    metadata.reflect(bind=engine)
    return engine, metadata.tables[table_name]


def convert_date_columns_to_str(df):
    if "start_date" in df.columns:
        df["start_date_timestamp"] = df.apply(lambda row: row.start_date.value, axis=1)
    if "end_date" in df.columns:
        df["end_date_timestamp"] = df.apply(lambda row: row.end_date.value, axis=1)
    return df


def get_dynamic_records_data(query_type, data_dict, is_advanced=False, date_option=None):
    engine, PLATFORM_TABLE = get_table(query_type)
    KEYS = FULL_KEYS_DICT[query_type]
    start_date = None
    end_date = None
    query = select(PLATFORM_TABLE.columns)
    if "strategy_names" in data_dict:
        engine_s, STRATEGY_TABLE = get_table("strategy")
        strategy_query = select(STRATEGY_TABLE.columns.id)
        strategy_query = strategy_query.filter(STRATEGY_TABLE.columns.name.in_(data_dict["strategy_names"]))
        strategy_result = engine_s.execute(strategy_query).fetchall()
        strategy_id_list = [x[0] for x in strategy_result]
        if "strategy_ids" in data_dict.keys():
            combined_set = set()
            combined_set.update(data_dict["strategy_ids"])
            combined_set.update(strategy_id_list)
            data_dict["strategy_ids"] = list(combined_set)
        else:
            data_dict["strategy_ids"] = strategy_id_list
    if "position_ids" in data_dict:
        engine, TRANSACTION_TABLE = get_table("transaction")
        transaction_query = select(TRANSACTION_TABLE.columns)
        transaction_query = transaction_query.filter(TRANSACTION_TABLE.columns.position_id.in_(data_dict["position_ids"]))
        strategy_result = engine.execute(transaction_query).fetchall()
        strategy_result = pd.DataFrame(strategy_result, columns=FULL_KEYS_DICT[query_type])
        if not is_advanced:
            return convert_date_columns_to_str(strategy_result)
        return convert_date_columns_to_str(ADVANCED_DICT_FUNC[query_type](strategy_result))
    if "platform_ids" in data_dict:
        engine, STRATEGY_TABLE = get_table("strategy")
        strategy_query = select(STRATEGY_TABLE.columns.id)
        strategy_query = strategy_query.filter(STRATEGY_TABLE.columns.platform_id.in_(data_dict["platform_ids"]))
        strategy_result = engine.execute(strategy_query).fetchall()
        strategy_id_list = [x[0] for x in strategy_result]
        if query_type == "strategy":
            strategy_result = pd.DataFrame(strategy_result, columns=FULL_KEYS_DICT[query_type])
            if not is_advanced:
                return convert_date_columns_to_str(strategy_result)
            return convert_date_columns_to_str(ADVANCED_DICT_FUNC[query_type](strategy_result))
        list_of_strategy_id = [row[0] for row in strategy_result]
        if "strategy_ids" in data_dict.keys():
            set1 = set(data_dict["strategy_ids"])
            set2 = set(strategy_id_list)
            data_dict["strategy_ids"] = list(set1 & set2)
        else:
            data_dict["strategy_ids"] = strategy_id_list
    for key in data_dict.keys():
        if key == "coins":
            query = query.filter(getattr(getattr(PLATFORM_TABLE, "columns"), "symbol").in_(data_dict[key]))
            continue
        if key == "side":
            if data_dict[key].upper() == "BOTH":
                continue
        if key == "strategy_ids":
            query = query.filter(getattr(getattr(PLATFORM_TABLE, "columns"), "strategy_id").in_(data_dict[key]))
            continue
        if key not in KEYS:
            continue
        if key == "start_date":
            start_date = datetime.datetime.strptime(data_dict[key], "%Y-%m-%dT%H:%M:%S.%fZ")
            continue
        if key == "end_date":
            end_date = datetime.datetime.strptime(data_dict[key], "%Y-%m-%dT%H:%M:%S.%fZ")
            continue
        query = query.filter(getattr(getattr(PLATFORM_TABLE, "columns"), key) == data_dict[key])
    if query_type == "position":
        query = query.filter(PLATFORM_TABLE.columns["end_by"] != "")
    result = engine.execute(query).fetchall()
    result = pd.DataFrame(result, columns=FULL_KEYS_DICT[query_type])
    if not date_option or date_option == '1':
        if start_date:
            result = result.loc[result.end_date >= start_date]
        if end_date:
            result = result.loc[result.end_date <= end_date]
    else:
        if start_date:
            result = result.loc[result.start_date >= start_date]
        if end_date:
            result = result.loc[result.start_date <= end_date]
    if result.empty:
        return result
    if not is_advanced:
        return convert_date_columns_to_str(result)
    return convert_date_columns_to_str(ADVANCED_DICT_FUNC[query_type](result)).fillna(-1)


def get_live_positions_from_db():
    query_type = "position"
    engine, POSITION_TABLE = get_table(query_type)
    query = select(POSITION_TABLE.columns).filter(POSITION_TABLE.columns["end_by"] == "")
    result = engine.execute(query).fetchall()
    result = pd.DataFrame(result, columns=FULL_KEYS_DICT[query_type])
    if result.empty:
        return result
    return convert_date_columns_to_str(ADVANCED_DICT_FUNC["live"](result)).fillna(-1)


def get_live_manually_operation_from_db(position_id=None):
    query_type = "manually_operation"
    engine, MANUALLY_TABLE = get_table(query_type)
    query = select(MANUALLY_TABLE.columns)
    if position_id:
        regex1 = f"%'position_id': {position_id},%"
        regex2 = f"%'position_id': {position_id}" + "}%"
        query = query.filter(or_(MANUALLY_TABLE.columns.params.ilike(regex1), MANUALLY_TABLE.columns.params.ilike(regex2)))
    result = engine.execute(query).fetchall()
    result = pd.DataFrame(result, columns=FULL_KEYS_DICT[query_type])

    result = result.iloc[::-1]
    return result.head(500)


def get_position_data(position_id):
    engine, PLATFORM_TABLE = get_table("position")
    query = select(PLATFORM_TABLE.columns).filter(getattr(getattr(PLATFORM_TABLE, "columns"), "id") == position_id)
    position_result = engine.execute(query).fetchall()
    position_result = pd.DataFrame(position_result, columns=FULL_KEYS_DICT["position"])
    end_by = position_result.end_by.iloc[0]
    if end_by == "":
        position_result = convert_date_columns_to_str(ADVANCED_DICT_FUNC["live"](position_result))
    else:
        position_result = convert_date_columns_to_str(ADVANCED_DICT_FUNC["position"](position_result))

    engine, PLATFORM_TABLE = get_table("transaction")
    query = select(PLATFORM_TABLE.columns).filter(getattr(getattr(PLATFORM_TABLE, "columns"), "position_id") == position_id)
    transaction_result = engine.execute(query).fetchall()
    transaction_result = pd.DataFrame(transaction_result, columns=FULL_KEYS_DICT["transaction"])
    transaction_result = convert_date_columns_to_str(ADVANCED_DICT_FUNC["transaction"](transaction_result))

    return {"position": position_result.fillna(-1), "transaction": transaction_result}


def get_strategy_data(strategy_id, advanced):
    engine, PLATFORM_TABLE = get_table("strategy")
    query = select(PLATFORM_TABLE.columns).filter(getattr(getattr(PLATFORM_TABLE, "columns"), "id") == strategy_id)
    strategy_result = engine.execute(query).fetchall()
    strategy_result = pd.DataFrame(strategy_result, columns=FULL_KEYS_DICT["strategy"])
    if advanced:
        strategy_result = convert_date_columns_to_str(ADVANCED_DICT_FUNC["strategy"](strategy_result))

    strategy_result = convert_date_columns_to_str(strategy_result)
    return {"strategy": strategy_result}


def get_platform_data(platform_id, advanced):
    engine, PLATFORM_TABLE = get_table("platform")
    query = select(PLATFORM_TABLE.columns).filter(getattr(getattr(PLATFORM_TABLE, "columns"), "id") == platform_id)
    platform_result = engine.execute(query).fetchall()
    platform_result = pd.DataFrame(platform_result, columns=FULL_KEYS_DICT["platform"])

    engine, STRATEGY_TABLE = get_table("strategy")
    query = select(STRATEGY_TABLE.columns).filter(getattr(getattr(STRATEGY_TABLE, "columns"), "platform_id") == platform_id)
    strategy_result = engine.execute(query).fetchall()
    strategy_list = [row[0] for row in strategy_result]

    platform_result = platform_result.assign(strategy_list=pd.Series([str(strategy_list)]).values)
    if advanced:
        platform_result = ADVANCED_DICT_FUNC["platform"](platform_result)
    # platform_result = convert_date_columns_to_str(platform_result)
    return {"platform": platform_result}


def create_transaction_transaction_df(transaction_df):
    return transaction_df


def create_position_statistics_df(position_df):
    transaction_df = get_dynamic_records_data("transaction", {"position_ids": list(position_df.id)}, is_advanced=True)
    for single_position in position_df.itertuples():
        create_single_position(position_df, single_position, transaction_df)
    position_df = position_df.sort_values(by=['id'], ascending=False)
    return position_df


def create_avg_buy_by_type(transaction_df, c_type):
    type_df = transaction_df.loc[transaction_df.type == c_type].copy()
    total_amount_buy = type_df.amount_stable.sum()
    type_df["percent_from_all"] = np.abs(type_df.amount_stable / total_amount_buy)
    type_df["part_to_avg"] = np.multiply(type_df["percent_from_all"], type_df.price)
    type_avg = type_df["part_to_avg"].sum()
    return type_avg


def create_single_position(position_df, single_position, transaction_df):
    id = single_position.id
    transaction_df = transaction_df.loc[transaction_df.position_id == id].copy()
    transaction_df['amount_stable'] = np.where(transaction_df.type == "SELL",
                                               np.multiply(transaction_df.amount, transaction_df.price),
                                               -1 * np.multiply(transaction_df.amount, transaction_df.price))
    change_amout = transaction_df.amount_stable.sum()
    position_type = transaction_df.iloc[0].type
    position_df.loc[position_df.id == id, 'change_amount'] = change_amout
    position_df.loc[position_df.id == id, 'duration'] = str(
        position_df.loc[position_df.id == id].iloc[0].end_date - position_df.loc[position_df.id == id].iloc[0].start_date)
    position_df.loc[position_df.id == id, 'change_amount_fee'] = transaction_df.amount_stable.sum() - transaction_df.amount_fee.sum()
    position_df.loc[position_df.id == id, 'change_percent'] = np.divide(change_amout, np.abs(
        transaction_df.loc[transaction_df.type == position_type].amount_stable.sum()))
    position_df.loc[position_df.id == id, 'total_invest'] = np.absolute(
        np.abs(transaction_df.loc[transaction_df.type == position_type].amount_stable.sum()))
    position_df.loc[position_df.id == id, 'avg_buy_price'] = create_avg_buy_by_type(transaction_df, "BUY")
    position_df.loc[position_df.id == id, 'avg_sell_price'] = create_avg_buy_by_type(transaction_df, "SELL")
    position_df.loc[position_df.id == id, 'total_fees'] = transaction_df.amount_fee.sum()


def create_live_statistics_df(position_df):
    coin_list = list(set(position_df.symbol))
    stable = list(set(position_df.stable))
    if len(stable) > 1:
        print("Problem- to many stable")
    symbol_to_price_df = get_relevant_price_list(coin_list, stable[0])
    relative_end_time = datetime.datetime.now()
    relative_end_time = datetime.datetime(relative_end_time.year, relative_end_time.month, relative_end_time.day,
                                          relative_end_time.hour, relative_end_time.minute, 0)
    transaction_df = get_dynamic_records_data("transaction", {"position_ids": list(position_df.id)}, is_advanced=True)
    for i, single_position in enumerate(position_df.itertuples()):
        calculate_single_live(i, position_df, relative_end_time, single_position, symbol_to_price_df, transaction_df)
    position_df = position_df.sort_values(by=['id'], ascending=False)
    return position_df


def calculate_single_live(i, position_df, relative_end_time, single_position, symbol_to_price_df, transaction_df):
    id = single_position.id
    coin = single_position.symbol
    current_price = float(symbol_to_price_df.loc[symbol_to_price_df.coin == coin].price.iloc[0])
    position_df.at[i, "end_date"] = relative_end_time

    transaction_df = transaction_df.loc[transaction_df.position_id == id]
    transaction_df = add_transaction_df_to_live(transaction_df, symbol_to_price_df, coin)
    transaction_df['amount_stable'] = np.where(transaction_df.type == "SELL",
                                               np.multiply(transaction_df.amount, transaction_df.price),
                                               -1 * np.multiply(transaction_df.amount, transaction_df.price))
    change_amount_stable = transaction_df.amount_stable.sum()
    first_transaction_in_position = transaction_df.iloc[0].copy()
    position_type = first_transaction_in_position.type

    position_df.loc[position_df.id == id, 'current_price'] = current_price
    position_df.loc[position_df.id == id, 'amount_stable'] = np.absolute(np.multiply(current_price, single_position.amount))
    position_df.loc[position_df.id == id, 'total_invest'] = np.absolute(
        np.abs(transaction_df.loc[transaction_df.type == position_type].amount_stable.sum()))
    position_df.loc[position_df.id == id, 'change_amount'] = change_amount_stable
    position_df.loc[position_df.id == id, 'change_amount_fee'] = transaction_df.amount_stable.sum() - transaction_df.amount_fee.sum()
    position_df.loc[position_df.id == id, 'change_percent'] = np.divide(change_amount_stable, np.abs(
        transaction_df.loc[transaction_df.type == position_type].amount_stable.sum()))
    position_df.loc[position_df.id == id, 'avg_buy_price'] = create_avg_buy_by_type(transaction_df, "BUY")
    position_df.loc[position_df.id == id, 'avg_sell_price'] = create_avg_buy_by_type(transaction_df, "SELL")
    position_df.loc[position_df.id == id, 'total_fees'] = transaction_df.amount_fee.sum()

    current_position_data = position_df.loc[position_df.id == id].iloc[0]
    current_amount_stable = np.abs(current_position_data.amount_stable)
    if position_type == "SELL":
        stop_percent = np.divide(current_position_data.avg_sell_price - current_position_data.stop, current_position_data.avg_sell_price)
        position_df.loc[position_df.id == id, 'stop_percent'] = stop_percent
        position_df.loc[position_df.id == id, 'stop_amount'] = np.multiply(stop_percent, current_amount_stable)
        if current_position_data["take"]:
            take_percent = np.divide(current_position_data.avg_sell_price - current_position_data["take"], current_position_data.avg_sell_price)
        else:
            take_percent = 0
        position_df.loc[position_df.id == id, 'take_percent'] = take_percent
        position_df.loc[position_df.id == id, 'take_amount'] = np.multiply(take_percent, current_amount_stable)
    if position_type == "BUY":
        stop_percent = np.divide(current_position_data.stop - current_position_data.avg_buy_price, current_position_data.avg_buy_price)
        position_df.loc[position_df.id == id, 'stop_percent'] = stop_percent
        position_df.loc[position_df.id == id, 'stop_amount'] = np.multiply(stop_percent, current_amount_stable)
        if current_position_data["take"]:
            take_percent = np.divide(current_position_data["take"] - current_position_data.avg_buy_price, current_position_data.avg_buy_price)
        else:
            take_percent = 0
        position_df.loc[position_df.id == id, 'take_percent'] = take_percent
        position_df.loc[position_df.id == id, 'take_amount'] = np.multiply(take_percent, current_amount_stable)


def create_strategy_statistics_df(strategy_df):
    position_records = get_dynamic_records_data("position", {"strategy_ids": list(strategy_df.id)}, is_advanced=True)
    for single_strategy in strategy_df.itertuples():
        calculate_single_strategy(single_strategy, strategy_df, position_records)

    strategy_df.replace([np.inf, -np.inf], 0, inplace=True)
    strategy_df = strategy_df.sort_values(by=['id'], ascending=False)
    return strategy_df


def calculate_single_strategy(single_strategy, strategy_df, position_records):
    id = single_strategy.id
    position_records = position_records.loc[position_records.strategy_id == id].copy()
    strategy_df.loc[strategy_df.id == id, 'change_amount'] = position_records.change_amount.sum()
    position_records.duration = pd.to_timedelta(position_records.duration)
    strategy_df.loc[strategy_df.id == id, 'duration'] = position_records.duration.mean()
    position_records.duration = str(position_records.duration)
    strategy_df.loc[strategy_df.id == id, 'change_amount_fee'] = position_records.change_amount_fee.sum()
    strategy_df.loc[strategy_df.id == id, 'change_percent'] = position_records.change_percent.sum()
    strategy_df.loc[strategy_df.id == id, 'avg_change_amount'] = position_records.change_amount.mean()
    strategy_df.loc[strategy_df.id == id, 'avg_change_amount_fee'] = position_records.change_amount_fee.mean()
    strategy_df.loc[strategy_df.id == id, 'avg_change_percent'] = position_records.change_percent.mean()
    strategy_df.loc[strategy_df.id == id, 'all_position_count'] = len(position_records.index)
    strategy_df.loc[strategy_df.id == id, 'profit_position_count'] = len(position_records.loc[position_records.change_amount_fee > 0].index)
    strategy_df.loc[strategy_df.id == id, 'loss_position_count'] = len(position_records.loc[position_records.change_amount_fee < 0].index)
    strategy_df.loc[strategy_df.id == id, 'profit_position_sum'] = position_records.loc[
        position_records.change_amount_fee > 0].change_amount.sum()
    strategy_df.loc[strategy_df.id == id, 'loss_position_sum'] = position_records.loc[
        position_records.change_amount_fee < 0].change_amount.sum()
    strategy_df.loc[strategy_df.id == id, 'profit_factor'] = np.abs(np.divide(strategy_df.profit_position_sum, strategy_df.loss_position_sum))
    strategy_df.loc[strategy_df.id == id, 'win_percent'] = np.divide(strategy_df.profit_position_count, strategy_df.all_position_count)


def create_platform_statistics_df(platform_df):
    try:
        all_strategy_df = get_dynamic_records_data("strategy", {"platform_ids": list(platform_df.id)}, is_advanced=True)
        for single_platform in platform_df.itertuples():
            calculate_single_platform(platform_df, single_platform, all_strategy_df)
        platform_df = platform_df.sort_values(by=['id'], ascending=False)
    except Exception as e:
        print(f"Failed due to {e}")
    return platform_df


def calculate_single_platform(platform_df, single_platform, strategy_records):
    id = single_platform.id
    strategy_records = strategy_records.loc[strategy_records.platform_id == id]
    strategy_records.duration = pd.to_timedelta(strategy_records.duration)
    platform_df.loc[platform_df.id == id, 'duration'] = strategy_records.duration.mean()
    strategy_records.duration = str(strategy_records.duration)
    platform_df.loc[platform_df.id == id, 'strategy_id_list'] = str(list(strategy_records.id))
    platform_df.loc[platform_df.id == id, 'change_amount'] = strategy_records.change_amount.sum()
    platform_df.loc[platform_df.id == id, 'change_amount_fee'] = strategy_records.change_amount_fee.sum()
    platform_df.loc[platform_df.id == id, 'change_percent'] = strategy_records.change_percent.sum()
    platform_df.loc[platform_df.id == id, 'avg_change_amount'] = strategy_records.avg_change_amount.mean()
    platform_df.loc[platform_df.id == id, 'avg_change_percent'] = strategy_records.avg_change_percent.mean()
    platform_df.loc[platform_df.id == id, 'all_position_count'] = strategy_records.all_position_count.sum()
    platform_df.loc[platform_df.id == id, 'avg_change_amount_fee'] = np.divide(platform_df.change_amount_fee,
                                                                               platform_df.all_position_count)
    platform_df.loc[platform_df.id == id, 'profit_strategy_count'] = strategy_records.profit_position_count.sum()
    platform_df.loc[platform_df.id == id, 'loss_strategy_count'] = strategy_records.loss_position_count.sum()
    platform_df.loc[platform_df.id == id, 'profit_strategy_sum'] = strategy_records.profit_position_sum.sum()
    platform_df.loc[platform_df.id == id, 'loss_strategy_sum'] = strategy_records.loss_position_sum.sum()
    platform_df.loc[platform_df.id == id, 'profit_factor'] = strategy_records.profit_factor.mean()
    platform_df.loc[platform_df.id == id, 'win_percent'] = strategy_records.win_percent.mean()
    platform_df.replace([np.inf, -np.inf], 0, inplace=True)


def delete_transactions_by_position(position_id):
    engine, TRANSACTION_TABLE = get_table("transaction")
    update_query = delete(TRANSACTION_TABLE).where(and_(
        TRANSACTION_TABLE.c.position_id == position_id,
    ))

    with engine.connect() as connection:
        with connection.begin() as transaction:
            try:
                connection.execute(update_query)
            except:
                transaction.rollback()
                raise
            else:
                transaction.commit()
    return True


def delete_position_by_strategy(strategy_id):
    engine, POSITION_TABLE = get_table("position")
    update_query = delete(POSITION_TABLE).where(and_(
        POSITION_TABLE.c.strategy_id == strategy_id,
    ))

    with engine.connect() as connection:
        with connection.begin() as transaction:
            try:
                connection.execute(update_query)
            except:
                transaction.rollback()
                raise
            else:
                transaction.commit()
    return True


def delete_strategy_by_platform(platform_id):
    engine, STRATEGY_TABLE = get_table("strategy")
    update_query = delete(STRATEGY_TABLE).where(and_(
        STRATEGY_TABLE.c.platform_id == platform_id,
    ))

    with engine.connect() as connection:
        with connection.begin() as transaction:
            try:
                connection.execute(update_query)
            except:
                transaction.rollback()
                raise
            else:
                transaction.commit()
    return True


def delete_platform_by_id(platform_id):
    engine, PLATFORM_TABLE = get_table("platform")
    update_query = delete(PLATFORM_TABLE).where(and_(
        PLATFORM_TABLE.c.id == platform_id,
    ))

    with engine.connect() as connection:
        with connection.begin() as transaction:
            try:
                connection.execute(update_query)
            except:
                transaction.rollback()
                raise
            else:
                transaction.commit()
    return True


def delte_all_platform_ref(platform_id):
    strategies = get_dynamic_records_data("strategy", {"platform_id": platform_id}, is_advanced=False)
    strategy_id_list = list(strategies.id)
    positions = get_dynamic_records_data("position", {"strategy_ids": strategy_id_list}, is_advanced=False)
    positions_id_list = list(positions.id)
    for position_id in positions_id_list:
        delete_transactions_by_position(position_id)
    for strategy_id in strategy_id_list:
        delete_position_by_strategy(strategy_id)
    delete_strategy_by_platform(platform_id)
    delete_platform_by_id(platform_id)


def create_statiscs_strategy_df2(position_df):
    all_strategy_ids = list(set(position_df.strategy_id))
    headers = ["id", "name", "symbols", "intervals", "positions", "profit", "win_percent", "profit_factor", "fees", "average_change",
               "min_leverage", "max_leverage", "json_format"]
    main_list = []
    for single_strategy_id in all_strategy_ids:
        part_df_by_strategy = position_df.loc[position_df.strategy_id == single_strategy_id]
        strategy_data = get_strategy_data(single_strategy_id, False)['strategy'].iloc[0]

        strategy_name = strategy_data["name"]
        intervals = strategy_data.intervals
        json_format = strategy_data.json_format

        total_fees = part_df_by_strategy.total_fees.sum()
        avg_change_percent = part_df_by_strategy.change_percent.mean()
        total_profit = part_df_by_strategy.change_amount_fee.sum()

        # part_df_by_strategy['duration_timestamp'] = pd.to_datetime(part_df_by_strategy['duration'])
        #
        # avg_time_for_pos = part_df_by_strategy.duration_timestamp.mean()
        total_positions = part_df_by_strategy.shape[0]
        symbol_we_run_on = ", ".join(list(set(part_df_by_strategy.symbol)))
        max_leverage = part_df_by_strategy.leverage.max()
        min_leverage = part_df_by_strategy.leverage.min()

        df_of_winners = part_df_by_strategy.loc[part_df_by_strategy.change_amount_fee >= 0]
        df_of_lossers = part_df_by_strategy.loc[part_df_by_strategy.change_amount_fee < 0]
        win_percent = df_of_winners.shape[0] / total_positions

        losser_sum = np.abs(df_of_lossers.change_amount_fee.sum())
        if losser_sum != 0:
            profit_factor = np.divide(
                df_of_winners.change_amount_fee.sum(),
                losser_sum)
        else:
            profit_factor = 0

        main_list.append([single_strategy_id, strategy_name, symbol_we_run_on, intervals, total_positions, total_profit, win_percent, profit_factor, total_fees, avg_change_percent,
                          min_leverage, max_leverage, json_format])
    df = pd.DataFrame(main_list, columns=headers)
    return df, headers


def weighted_average(column, weights):
    return np.average(column, weights=weights)


def unique_symbols(series):
    unique_symbols_list = []
    for symbols in series:
        unique_symbols_list.append(symbols)
    return list(set(unique_symbols_list))


def all_ids(series):
    return list(series)


def create_statiscs_strategy_df(position_df):
    all_strategy_ids = list(set(position_df.strategy_id))
    headers = ["id", "name", "symbols", "intervals", "positions", "duration", "profit", "win_percent", "profit_factor", "fees", "average_change",
               "min_leverage", "max_leverage", "json_format"]
    main_list = []
    for single_strategy_id in all_strategy_ids:
        part_df_by_strategy = position_df.loc[position_df.strategy_id == single_strategy_id]
        strategy_data = get_strategy_data(single_strategy_id, False)['strategy'].iloc[0]

        strategy_name = strategy_data["name"]
        intervals = strategy_data.intervals
        json_format = strategy_data.json_format

        total_fees = part_df_by_strategy.total_fees.sum()
        avg_change_percent = part_df_by_strategy.change_percent.mean()
        total_profit = part_df_by_strategy.change_amount_fee.sum()
        part_df_by_strategy['duration'] = pd.to_timedelta(part_df_by_strategy['duration'])
        duration = part_df_by_strategy.duration.mean()

        total_positions = part_df_by_strategy.shape[0]
        symbol_we_run_on = ", ".join(list(set(part_df_by_strategy.symbol)))
        max_leverage = part_df_by_strategy.leverage.max()
        min_leverage = part_df_by_strategy.leverage.min()

        df_of_winners = part_df_by_strategy.loc[part_df_by_strategy.change_amount_fee >= 0]
        df_of_lossers = part_df_by_strategy.loc[part_df_by_strategy.change_amount_fee < 0]
        win_percent = df_of_winners.shape[0] / total_positions

        losser_sum = np.abs(df_of_lossers.change_amount_fee.sum())
        if losser_sum != 0:
            profit_factor = np.divide(
                df_of_winners.change_amount_fee.sum(),
                losser_sum)
        else:
            profit_factor = 0

        main_list.append([single_strategy_id, strategy_name, symbol_we_run_on, intervals, total_positions, duration, total_profit, win_percent, profit_factor, total_fees, avg_change_percent,
                          min_leverage, max_leverage, json_format])
    statistics_df_by_id = pd.DataFrame(main_list, columns=headers)
    result_df = statistics_df_by_id.groupby('name').agg(
        id=('id', all_ids),
        symbols=('symbols', unique_symbols),
        intervals=('intervals', 'first'),
        positions=('positions', 'sum'),
        duration=('duration', "mean"),
        profit=('profit', 'sum'),
        win_percent=('win_percent', lambda x: weighted_average(x, statistics_df_by_id.loc[x.index, 'positions'])),
        profit_factor=('profit_factor', lambda x: weighted_average(x, statistics_df_by_id.loc[x.index, 'positions'])),
        fees=('fees', 'sum'),
        average_change=('positions', lambda x: weighted_average(x, statistics_df_by_id.loc[x.index, 'positions'])),
        min_leverage=('min_leverage', 'min'),
        max_leverage=('max_leverage', 'max'),
        json_format=('json_format', 'first'),
    ).reset_index()
    headers = ["name", "id", "symbols", "intervals", "positions", "duration", "profit", "win_percent", "profit_factor", "fees", "average_change",
               "min_leverage", "max_leverage", "json_format"]
    print("Create Statistics Has been Done!")
    return result_df, headers


def create_statiscs_strategy_coin(position_df):
    all_strategy_coins = list(set(position_df.symbol))
    headers = ["symbol", "positions", "duration", "profit", "win_percent", "profit_factor", "fees", "average_change",
               "min_leverage", "max_leverage"]
    main_list = []
    for single_symbol in all_strategy_coins:
        part_df_by_symbol = position_df.loc[position_df.symbol == single_symbol]
        # strategy_data = get_strategy_data(single_symbol, False)['strategy'].iloc[0]
        #
        # strategy_name = strategy_data["name"]
        # intervals = strategy_data.intervals
        # json_format = strategy_data.json_format

        total_fees = part_df_by_symbol.total_fees.sum()
        avg_change_percent = part_df_by_symbol.change_percent.mean()
        total_profit = part_df_by_symbol.change_amount_fee.sum()
        part_df_by_symbol['duration'] = pd.to_timedelta(part_df_by_symbol['duration'])
        duration = part_df_by_symbol.duration.mean()

        total_positions = part_df_by_symbol.shape[0]
        symbol_we_run_on = ", ".join(list(set(part_df_by_symbol.symbol)))
        max_leverage = part_df_by_symbol.leverage.max()
        min_leverage = part_df_by_symbol.leverage.min()

        df_of_winners = part_df_by_symbol.loc[part_df_by_symbol.change_amount_fee >= 0]
        df_of_lossers = part_df_by_symbol.loc[part_df_by_symbol.change_amount_fee < 0]
        win_percent = df_of_winners.shape[0] / total_positions

        losser_sum = np.abs(df_of_lossers.change_amount_fee.sum())
        if losser_sum != 0:
            profit_factor = np.divide(
                df_of_winners.change_amount_fee.sum(),
                losser_sum)
        else:
            profit_factor = 0

        main_list.append([single_symbol, total_positions, duration, total_profit, win_percent, profit_factor, total_fees, avg_change_percent, min_leverage, max_leverage])
    statistics_df_by_id = pd.DataFrame(main_list, columns=headers)
    result_df = statistics_df_by_id.groupby('symbol').agg(
        positions=('positions', 'sum'),
        duration=('duration', "mean"),
        profit=('profit', 'sum'),
        win_percent=('win_percent', lambda x: weighted_average(x, statistics_df_by_id.loc[x.index, 'positions'])),
        profit_factor=('profit_factor', lambda x: weighted_average(x, statistics_df_by_id.loc[x.index, 'positions'])),
        fees=('fees', 'sum'),
        average_change=('positions', lambda x: weighted_average(x, statistics_df_by_id.loc[x.index, 'positions'])),
        min_leverage=('min_leverage', 'min'),
        max_leverage=('max_leverage', 'max'),
    ).reset_index()
    headers = ["symbol", "positions", "duration", "profit", "win_percent", "profit_factor", "fees", "average_change", "min_leverage", "max_leverage"]
    print("Create Statistics Has been Done!")
    return result_df, headers


def create_statiscs_hour_date(position_df):
    headers = ["delta_time", "day", "hour", "symbols", "positions", "duration", "profit", "win_percent", "profit_factor", "fees", "average_change",
               "min_leverage", "max_leverage"]
    x = 1
    main_list = []
    while x < 169:
        delta_time_hour = x % 24
        delta_time_day = x // 24

        part_df_by_symbol = position_df.loc[(position_df['start_date'].dt.hour == delta_time_hour) & (position_df['start_date'].dt.weekday == delta_time_day)]

        if part_df_by_symbol.empty:
            x += 1
            continue
        total_fees = part_df_by_symbol.total_fees.sum()
        avg_change_percent = part_df_by_symbol.change_percent.mean()
        total_profit = part_df_by_symbol.change_amount_fee.sum()
        part_df_by_symbol['duration'] = pd.to_timedelta(part_df_by_symbol['duration'])
        duration = part_df_by_symbol.duration.mean()

        total_positions = part_df_by_symbol.shape[0]
        symbol_we_run_on = ", ".join(list(set(part_df_by_symbol.symbol)))
        max_leverage = part_df_by_symbol.leverage.max()
        min_leverage = part_df_by_symbol.leverage.min()

        df_of_winners = part_df_by_symbol.loc[part_df_by_symbol.change_amount_fee >= 0]
        df_of_lossers = part_df_by_symbol.loc[part_df_by_symbol.change_amount_fee < 0]
        if total_positions > 0:
            win_percent = df_of_winners.shape[0] / total_positions
        else:
            win_percent = 0

        losser_sum = np.abs(df_of_lossers.change_amount_fee.sum())
        if losser_sum != 0:
            profit_factor = np.divide(
                df_of_winners.change_amount_fee.sum(),
                losser_sum)
        else:
            profit_factor = 0

        main_list.append(
            [datetime.timedelta(hours=x), delta_time_day + 1, delta_time_hour, symbol_we_run_on, total_positions, duration, total_profit, win_percent, profit_factor, total_fees, avg_change_percent,
             min_leverage, max_leverage])
        x += 1
    statistics_df_by_id = pd.DataFrame(main_list, columns=headers)
    result_df = statistics_df_by_id.groupby('delta_time').agg(
        day=('day', "first"),
        hour=('hour', 'first'),
        symbols=('symbols', unique_symbols),
        positions=('positions', 'sum'),
        duration=('duration', "mean"),
        profit=('profit', 'sum'),
        win_percent=('win_percent', lambda x: weighted_average(x, statistics_df_by_id.loc[x.index, 'positions'])),
        profit_factor=('profit_factor', lambda x: weighted_average(x, statistics_df_by_id.loc[x.index, 'positions'])),
        fees=('fees', 'sum'),
        average_change=('positions', lambda x: weighted_average(x, statistics_df_by_id.loc[x.index, 'positions'])),
        min_leverage=('min_leverage', 'min'),
        max_leverage=('max_leverage', 'max'),
    ).reset_index()
    result_df = result_df.sort_values(by=["delta_time"])
    headers = ["delta_time", "day", "hour", "symbols", "positions", "duration", "profit", "win_percent", "profit_factor", "fees", "average_change", "min_leverage", "max_leverage"]
    print("Create Statistics Has been Done!")
    return result_df, headers


ADVANCED_DICT_FUNC = {
    "platform": create_platform_statistics_df,
    "strategy": create_strategy_statistics_df,
    "position": create_position_statistics_df,
    "transaction": create_transaction_transaction_df,
    "live": create_live_statistics_df,
}
