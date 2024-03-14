import json
import json
import mimetypes
import os

import pandas as pd
from django.http import HttpResponse, Http404
from rest_framework.decorators import api_view

from configuration.my_logger import logger
from src.api.binance_api import get_future_balance_update, get_future_open_positions
from src.api.db_api import get_dynamic_records_data, get_position_data, get_strategy_data, get_platform_data, get_live_positions_from_db, \
    delte_all_platform_ref, get_live_manually_operation_from_db, create_statiscs_strategy_df, create_statiscs_strategy_coin, create_statiscs_hour_date
from src.constant import custom_serializer
from src.pages.position_records import get_position_and_basic_analysis, graph_function_to_type
from src.utils.generic_utils import get_bars_per_position_live, get_bars_per_position_done
from src.utils.mode_singelton import ModeSingleton

mode_object = ModeSingleton()


def check(request):
    return HttpResponse("OK")


@api_view(['POST'])
def get_db_dynamic(request):
    try:
        data_dict = request.data
        table_type = data_dict["type"]
        is_advanced = data_dict["advanced"]
        params = data_dict["params"]
        mode_object.set_mode(data_dict["mode"])
        date_option = None
        if "date_option" in params.keys():
            date_option = params["date_option"]
        answer_df = get_dynamic_records_data(table_type, params, is_advanced=is_advanced, date_option=date_option)
        answer_as_dict = answer_df.to_dict('records')
        answer_as_json = json.dumps(answer_as_dict, sort_keys=True, indent=4, default=str)
    except Exception as e:
        print(e)
        return Http404(f"Bad Request : {e}")

    return HttpResponse(answer_as_json, content_type="application/json")


@api_view(['POST'])
def position_records(request):
    logger.info("Position Records Start")
    data_dict = request.data
    mode_object.set_mode(data_dict["mode"])
    answer_as_dict = get_position_and_basic_analysis(data_dict["query"], data_dict["refresh"]).copy()
    logger.info("Got Basic Analyze")
    graph_interval = data_dict.get("graph_interval")
    graph_type = data_dict.get("graph_type")
    graph_by = data_dict.get("graph_by")
    only_graph = data_dict.get("only_graph", False)
    logger.info("Got Graph Time")
    if graph_by and graph_type and graph_interval:
        answer_as_dict["graph_data"] = graph_function_to_type[graph_type](answer_as_dict["positions"], graph_interval, graph_by)
    for key, value in answer_as_dict.items():
        if isinstance(value, pd.DataFrame):
            answer_as_dict[key] = value.to_dict("records")
    if only_graph:
        answer_as_dict = {"graph_data":answer_as_dict["graph_data"]}
    final_answer = json.dumps(answer_as_dict, default=custom_serializer)
    return HttpResponse(final_answer, content_type="application/json")


@api_view(['POST'])
def get_csv_records(request):
    try:
        data_dict = request.data
        table_type = data_dict["type"]
        is_advanced = data_dict["advanced"]
        params = data_dict["params"]
        mode_object.set_mode(data_dict["mode"])
        date_option = None
        if "date_option" in params.keys():
            date_option = params["date_option"]
        answer_df = get_dynamic_records_data(table_type, params, is_advanced=is_advanced, date_option=date_option)
        answer_df, headers = create_statiscs_strategy_df(answer_df)
        # answer_as_dict = answer_df.to_dict('records')
        # answer_as_json = json.dumps(answer_as_dict, sort_keys=True, indent=4, default=str)
        filename = 'output.csv'
        answer_df.to_csv(filename, index=False, header=headers)
        with open(filename, "rb") as f:
            name_to_return = f.name.split("/")[-1]
            content_type, encoding = mimetypes.guess_type(name_to_return)
            if not content_type:
                content_type = "application/octet-stream"
            file_response = HttpResponse(f.read(), content_type=content_type)
            file_response["Content-Type"] = content_type
            file_response["Content-Disposition"] = f'attachment; filename="{name_to_return}"'
        os.remove(filename)
        return file_response
    except Exception as e:
        print(e)
        return Http404(f"Bad Request : {e}")


@api_view(['POST'])
def get_csv_coin_records(request):
    try:
        data_dict = request.data
        table_type = data_dict["type"]
        is_advanced = data_dict["advanced"]
        params = data_dict["params"]
        mode_object.set_mode(data_dict["mode"])
        date_option = None
        if "date_option" in params.keys():
            date_option = params["date_option"]
        answer_df = get_dynamic_records_data(table_type, params, is_advanced=is_advanced, date_option=date_option)
        answer_df, headers = create_statiscs_strategy_coin(answer_df)
        filename = 'output.csv'
        answer_df.to_csv(filename, index=False, header=headers)
        with open(filename, "rb") as f:
            name_to_return = f.name.split("/")[-1]
            content_type, encoding = mimetypes.guess_type(name_to_return)
            if not content_type:
                content_type = "application/octet-stream"
            file_response = HttpResponse(f.read(), content_type=content_type)
            file_response["Content-Type"] = content_type
            file_response["Content-Disposition"] = f'attachment; filename="{name_to_return}"'
        os.remove(filename)
        return file_response
    except Exception as e:
        print(e)
        return Http404(f"Bad Request : {e}")


@api_view(['POST'])
def get_statiscs_hour_date(request):
    try:
        data_dict = request.data
        table_type = data_dict["type"]
        is_advanced = data_dict["advanced"]
        params = data_dict["params"]
        mode_object.set_mode(data_dict["mode"])
        date_option = None
        if "date_option" in params.keys():
            date_option = params["date_option"]
        answer_df = get_dynamic_records_data(table_type, params, is_advanced=is_advanced, date_option=date_option)
        answer_df, headers = create_statiscs_hour_date(answer_df)
        filename = 'output.csv'
        answer_df.to_csv(filename, index=False, header=headers)
        with open(filename, "rb") as f:
            name_to_return = f.name.split("/")[-1]
            content_type, encoding = mimetypes.guess_type(name_to_return)
            if not content_type:
                content_type = "application/octet-stream"
            file_response = HttpResponse(f.read(), content_type=content_type)
            file_response["Content-Type"] = content_type
            file_response["Content-Disposition"] = f'attachment; filename="{name_to_return}"'
        os.remove(filename)
        return file_response
    except Exception as e:
        print(e)
        return Http404(f"Bad Request : {e}")


@api_view(['POST'])
def get_poistion_analyze(request):
    try:
        data_dict = request.data
        position_id = data_dict["position_id"]
        interval = data_dict["interval"]
        bars = data_dict["bars"]
        mode_object.set_mode(data_dict["mode"])
        position_and_transactions = get_position_data(position_id)
        position_and_transactions["position"] = position_and_transactions["position"].to_dict('records')[0]
        position_and_transactions["transaction"] = position_and_transactions["transaction"].to_dict('records')
        position_and_transactions["bars"] = get_bars_per_position_done(position_and_transactions["position"]["symbol"],
                                                                       position_and_transactions["position"]["stable"], interval, bars,
                                                                       position_and_transactions["position"]["start_date"],
                                                                       position_and_transactions["position"]["end_date"]).to_dict('records')
        position_and_transactions["manually"] = get_live_manually_operation_from_db(position_id=position_id).to_dict('records')
        answer_as_json = json.dumps(position_and_transactions, sort_keys=True, indent=4, default=str)
    except Exception as e:
        print(e)
        return Http404(f"Bad Request : {e}")

    return HttpResponse(answer_as_json, content_type="application/json")


@api_view(['POST'])
def get_future_balance_analyze(request):
    try:
        # data_dict = request.data
        stable_coin = "USDT"
        data_for_future_balance = {}
        total_worth, only_active_asset = get_future_balance_update(stable_coin)
        live_positions = get_future_open_positions()
        data_for_future_balance["assets"] = only_active_asset.to_dict('records')
        data_for_future_balance["positions"] = live_positions.to_dict('records')
        data_for_future_balance["total_worth"] = total_worth
        answer_as_json = json.dumps(data_for_future_balance, sort_keys=True, indent=4, default=str)
    except Exception as e:
        print(e)
        return Http404(f"Bad Request : {e}")

    return HttpResponse(answer_as_json, content_type="application/json")


@api_view(['POST'])
def get_position_bars(request):
    try:
        data_dict = request.data
        symbol = data_dict["symbol"]
        stable = data_dict["stable"]
        interval = data_dict["interval"]
        start_date = data_dict["start_date"] / 1000000000
        bars = data_dict["bars"]
        mode_object.set_mode(data_dict["mode"])
        position_and_transactions = get_bars_per_position_live(symbol, stable, interval, bars, start_date).to_dict('records')
        answer_as_json = json.dumps(position_and_transactions, sort_keys=True, indent=4, default=str)
    except Exception as e:
        print(e)
        return Http404(f"Bad Request : {e}")

    return HttpResponse(answer_as_json, content_type="application/json")


@api_view(['POST'])
def get_strategy_analyze(request):
    try:
        data_dict = request.data
        mode_object.set_mode(data_dict["mode"])
        strategy_id = data_dict["strategy_id"]
        advanced = data_dict["advanced"]
        position_and_transactions = get_strategy_data(strategy_id, advanced)
        position_and_transactions["strategy"] = position_and_transactions["strategy"].to_dict('records')[0]
        answer_as_json = json.dumps(position_and_transactions, sort_keys=True, indent=4, default=str)
    except Exception as e:
        print(e)
        return Http404(f"Bad Request : {e}")

    return HttpResponse(answer_as_json, content_type="application/json")


@api_view(['POST'])
def get_platform_analyze(request):
    try:
        data_dict = request.data
        mode_object.set_mode(data_dict["mode"])
        platform_id = data_dict["platform_id"]
        advanced = data_dict["advanced"]
        position_and_transactions = get_platform_data(platform_id, advanced)
        position_and_transactions["platform"] = position_and_transactions["platform"].to_dict('records')[0]
        answer_as_json = json.dumps(position_and_transactions, sort_keys=True, indent=4, default=str)
    except Exception as e:
        print(e)
        return Http404(f"Bad Request : {e}")

    return HttpResponse(answer_as_json, content_type="application/json")


@api_view(['POST'])
def get_live_positions(request):
    try:
        data_dict = request.data
        mode_object.set_mode(data_dict["mode"])
        answer_df = get_live_positions_from_db()
        answer_as_dict = answer_df.to_dict('records')
        answer_as_json = json.dumps(answer_as_dict, sort_keys=True, indent=4, default=str)
    except Exception as e:
        return Http404(f"Bad Request : {e}")

    return HttpResponse(answer_as_json, content_type="application/json")


@api_view(['POST'])
def get_manually_operation(request):
    try:
        data_dict = request.data
        mode_object.set_mode(data_dict["mode"])
        answer_df = get_live_manually_operation_from_db()
        answer_as_dict = answer_df.to_dict('records')
        answer_as_json = json.dumps(answer_as_dict, sort_keys=True, indent=4, default=str)
    except Exception as e:
        return Http404(f"Bad Request : {e}")

    return HttpResponse(answer_as_json, content_type="application/json")


@api_view(['POST'])
def delete_platform_by_id(request):
    try:
        data_dict = request.data
        mode_object.set_mode(data_dict["mode"])
        platform_id = data_dict["platform_id"]
        delte_all_platform_ref(platform_id)
    except Exception as e:
        return Http404(f"Bad Request : {e}")

    return HttpResponse(status=200)
