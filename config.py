import os

API_VERSION = "0.1.0"
LOG_DIR = "logs"
DATA_DIR = "data"
OUTPUT_DIR = "outputs"
MAX_TEXT_LENGTH = 2000
RULE_BASED_SYSTEM = "rule-based"
LLM_BASED_SYSTEM = "llm-based"
DEFAULT_SYSTEM = RULE_BASED_SYSTEM
SYSTEMS_LIST = [RULE_BASED_SYSTEM, LLM_BASED_SYSTEM]
CORRECTION_LLM_NAME = "gpt-4.1-2025-04-14"
CORRECTION_LLM_MAX_TOKENS = 600
EDIT_LLM_NAME = "gpt-4.1-2025-04-14"
EDIT_LLM_MAX_TOKENS = 300
FEEDBACK_LLM_NAME = "gpt-4.1-2025-04-14"
FEEDBACK_LLM_MAX_TOKENS = 800
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS", "http://localhost:5173,http://localhost"
).split(",")
ENABLE_FILE_LOG = os.getenv("ENABLE_FILE_LOGGING", "false").lower() == "true"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-1")
SAVE_TO_S3 = os.getenv("SAVE_TO_S3", "false").lower() == "true"
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
ENABLE_FILE_LOGGING = os.getenv("ENABLE_FILE_LOGGING", "false").lower() == "true"
