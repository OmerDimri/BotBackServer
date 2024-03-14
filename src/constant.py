from datetime import datetime
from pandas import Timedelta

FULL_KEYS_DICT = {
    "platform": ["id", "name", "type", "status", "coins", "start_date"],
    "strategy": ["id", "platform_id", "name", "json_format", "intervals", "strategy_type"],
    "position": ["id", "strategy_id", "symbol", "stable", "amount", "side", "leverage", "end_by", "start_date", "end_date", "stop", "take", "update_sltp_date", "comment"],
    "transaction": ["id", "position_id", "symbol", "stable", "type", "price", "amount", "amount_fee", "date", "is_broker"],
    "manually_operation": ["id", "name", "params", "date"]
}


def custom_serializer(obj):
    if isinstance(obj, datetime):
        obj_as_str = str(obj)
        str_as_list = obj_as_str.split(":")[:-1]
        return ":".join(str_as_list)
    if isinstance(obj, Timedelta):
        obj_as_str = str(obj)
        str_as_list = obj_as_str.split(":")[:-1]
        return ":".join(str_as_list)
    else:
        raise TypeError("Type not serializable")
