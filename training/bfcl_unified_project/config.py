from pathlib import Path

VERSION_PREFIX = "BFCL_v4"

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data"
PROMPT_PATH = DATA_PATH
POSSIBLE_ANSWER_PATH = DATA_PATH / "possible_answer"
MULTI_TURN_FUNC_DOC_PATH = DATA_PATH / "multi_turn_func_doc"

LOCAL_SERVER_PORT = 8000

MULTI_TURN_CATEGORIES = [
    "multi_turn_base",
    "multi_turn_miss_param",
    "multi_turn_miss_func",
    "multi_turn_long_context",
]

MODEL_CONFIG = {
    "Qwen/Qwen3-8B-FC": {
        "model_name": "Qwen/Qwen3-8B",
        "is_fc_model": True,
    }
}

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
    "GorillaFileSystem": "func_source_code.gorilla_file_system",
    "MathAPI": "func_source_code.math_api",
    "MessageAPI": "func_source_code.message_api",
    "TwitterAPI": "func_source_code.posting_api",
    "TicketAPI": "func_source_code.ticket_api",
    "TradingBot": "func_source_code.trading_bot",
    "TravelAPI": "func_source_code.travel_booking",
    "VehicleControlAPI": "func_source_code.vehicle_control",
}

STATELESS_CLASSES = [
    "MathAPI",
]

MAXIMUM_STEP_LIMIT = 20

DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC = (
    "I have updated some more functions you can choose from. What about now?"
)

DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_PROMPTING = (
    "{functions}\n" + DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC
)
