FULL_KEYS_DICT = {
    "platform": ["id", "name", "type", "status", "coins", "start_date"],
    "strategy": ["id", "platform_id", "name", "json_format", "intervals", "strategy_type"],
    "position": ["id", "strategy_id", "symbol", "stable", "amount", "side", "leverage", "end_by", "start_date", "end_date", "stop", "take", "update_sltp_date", "comment"],
    "transaction": ["id", "position_id", "symbol", "stable", "type", "price", "amount", "amount_fee", "date", "is_broker"],
    "manually_operation": ["id", "name", "params", "date"]
}

