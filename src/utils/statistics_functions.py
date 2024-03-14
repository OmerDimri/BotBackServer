import numpy as np
import pandas as pd

PERCENT_COLUMNS = ["strategy_name", "symbol", "side", "end_by"]


def analyze_basic_statistics(position_df):
    #TODO: FIX THIS SHIT LAZY WORK
    position_count = position_df.shape[0]
    profit_factor, win_position_count = calculate_profit_factor(position_count, position_df)
    basic_statistic_dict = {
        "number_of_positions": len(position_df),
        "first_position_date": position_df.start_date.min(),
        "last_position_date": position_df.start_date.max(),
        "win_percent": win_position_count / position_count,
        "total_profit": position_df.profit.sum(),
        "total_fees": position_df.fee_stable.sum(),
        "avg_change": position_df.change_percent.mean() * 100,
        "avg_duration": position_df.duration.mean(),
        "profit_factor": profit_factor
    }

    for percent_column in PERCENT_COLUMNS:
        x = (position_df[percent_column].value_counts(normalize=True) * 100).round(2).to_dict()
        basic_statistic_dict[f"{percent_column}_percent_dict"] = list()
        for key, value in x.items():
            basic_statistic_dict[f"{percent_column}_percent_dict"].append({
                "type": key,
                "value": value
            })

    return basic_statistic_dict


def calculate_profit_factor(position_count, position_df):
    win_position_count = len(position_df[position_df['profit'] > 0])
    loose_position_count = len(position_df[position_df['profit'] <= 0])
    if position_count == 0 or win_position_count == 0 or loose_position_count == 0:
        profit_factor = 0
    else:
        avg_winner = np.divide(
            position_df.loc[position_df.profit > 0].profit.sum(),
            win_position_count
        )
        avg_lose = np.divide(
            position_df.loc[position_df.profit < 0].profit.sum(),
            loose_position_count
        )
        profit_factor = np.abs(np.divide(avg_winner, avg_lose))
    return profit_factor, win_position_count


def line_function(position_df, graph_interval, graph_by):
    if "m" in graph_interval:
        graph_interval.replace("m", "T")
    graph_interval = graph_interval.upper()
    if graph_by == "total":
        position_df_date = position_df.copy().set_index("start_date")
        result = position_df_date.resample(graph_interval).agg({"profit": 'sum'})
        result = result[result['profit'] != 0]
        result["profit"] = result.profit.cumsum().round(2)
        result["category"] = "total"
        result["date"] = result.index
        result = result.reset_index(drop=True)
        main_df = result
    else:
        optional_graph_by_values = list(set(position_df[graph_by]))
        main_df = pd.DataFrame()
        for graph_by_value in optional_graph_by_values:
            position_graph_by = position_df.loc[position_df[graph_by] == graph_by_value].copy().set_index("start_date")
            result = position_graph_by.resample(graph_interval).agg({"profit": 'sum'})
            result = result[result['profit'] != 0]
            result["profit"] = result.profit.cumsum().round(2)
            result["category"] = graph_by_value
            result["date"] = result.index
            result = result.reset_index(drop=True)
            if main_df.empty:
                main_df = result
            else:
                main_df = pd.concat([main_df, result], ignore_index=True)
    main_df = main_df.sort_values(by='date')
    return main_df.to_dict("records")


def bar_function(position_df, graph_interval, graph_by):
    if "m" in graph_interval:
        graph_interval.replace("m", "T")
    graph_interval = graph_interval.upper()
    if graph_by == "total":
        position_df_date = position_df.copy().set_index("start_date")
        result = position_df_date.resample(graph_interval).agg({"profit": 'sum'})
        result = result[result['profit'] != 0]
        result["profit"] = result.profit.round(2)
        result["date"] = result.index
        result = result.reset_index(drop=True)
        main_df = result
    else:
        optional_graph_by_values = list(set(position_df[graph_by]))
        main_df = pd.DataFrame()
        for graph_by_value in optional_graph_by_values:
            result = position_df.loc[position_df[graph_by] == graph_by_value].copy()
            result["profit"] = result.profit.sum().round(2)
            result["date"] = graph_by_value
            result = result.reset_index(drop=True)
            if main_df.empty:
                main_df = result
            else:
                main_df = pd.concat([main_df, result], ignore_index=True)
    main_df = main_df.sort_values(by='date')
    return main_df.to_dict("records")
