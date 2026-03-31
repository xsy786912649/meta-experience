from pathlib import Path

VERSION_PREFIX = "BFCL_v4"

PACKAGE_ROOT = Path(__file__).resolve().parent
DATA_PATH = PACKAGE_ROOT / "data"
PROMPT_PATH = DATA_PATH
POSSIBLE_ANSWER_PATH = DATA_PATH / "possible_answer"
MULTI_TURN_FUNC_DOC_PATH = DATA_PATH / "multi_turn_func_doc"

MULTI_TURN_CATEGORIES = [
    "multi_turn_base",
    "multi_turn_miss_param",
    "multi_turn_miss_func",
    "multi_turn_long_context",
]

MULTI_TURN_FUNC_DOC_FILE_MAPPING = {
    "GorillaFileSystem": "gorilla_file_system.json",
    "MathAPI": "math_api.json",
    "MessageAPI": "message_api.json",
    "TwitterAPI": "posting_api.json",
    "TicketAPI": "ticket_api.json",
    "TradingBot": "trading_bot.json",
    "TravelAPI": "travel_booking.json",
    "VehicleControlAPI": "vehicle_control.json",
}

CLASS_FILE_PATH_MAPPING = {
    "GorillaFileSystem": "recipe.bfcl_multiturn.func_source_code.gorilla_file_system",
    "MathAPI": "recipe.bfcl_multiturn.func_source_code.math_api",
    "MessageAPI": "recipe.bfcl_multiturn.func_source_code.message_api",
    "TwitterAPI": "recipe.bfcl_multiturn.func_source_code.posting_api",
    "TicketAPI": "recipe.bfcl_multiturn.func_source_code.ticket_api",
    "TradingBot": "recipe.bfcl_multiturn.func_source_code.trading_bot",
    "TravelAPI": "recipe.bfcl_multiturn.func_source_code.travel_booking",
    "VehicleControlAPI": "recipe.bfcl_multiturn.func_source_code.vehicle_control",
}

STATELESS_CLASSES = [
    "MathAPI",
]

MAXIMUM_STEP_LIMIT = 50
