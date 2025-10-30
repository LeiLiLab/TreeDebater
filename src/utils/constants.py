import json
import os

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__)) + "/../.."

####################### Constants #######################

MAX_TRY_NUM = 3
EMBEDDING_MODEL = "models/text-embedding-004"
SUPPORT_RM_PATH = os.path.join(WORKSPACE_DIR, "checkpoints/reward_pro")
ATTACK_RM_PATH = os.path.join(WORKSPACE_DIR, "checkpoints/reward_con")

####################### API Keys #######################

KEY_FILE = os.path.join(WORKSPACE_DIR, "src/configs", "api_key.json")

if os.path.exists(KEY_FILE):
    print(f"Loading API keys from {KEY_FILE}")
    with open(KEY_FILE, "r") as f:
        load_keys = json.load(f)
        for key in load_keys:
            if key not in os.environ:
                print(f"Warning: {key} not found in environment variables, loading from {KEY_FILE}")
                os.environ[key] = load_keys[key]
            else:
                print(f"Warning: {key} found in environment variables, skipping")
else:
    print(f"API keys not found in {KEY_FILE}")

openai_api_key = os.environ["OPENAI_API_KEY"]
google_api_key = os.environ["GOOGLE_API_KEY"]
togetherai_api_key = os.environ["TOGETHERAI_API_KEY"]
deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY", "")  # Get the DeepSeek API key if available


####################### Time Estimation #######################

WORD_BUDGET_FOR_DRAFT = 500
LENGTH_MODE_FOR_DRAFT = "phonemes"  # rough estimatation, phonemes/words/syllables
TIME_MODE_FOR_STATEMENT = "fastspeech"  # precise estimatation, fastspeech/openai
TIME_TOLERANCE = 15  # seconds

OPENING_TIME = REBUTTAL_TIME = 240
CLOSING_TIME = 120
DEFAULT_MAX_WORDS = 520

WORDRATIO = {"phonemes": 4.5, "words": 1, "syllables": 1.75, "fastspeech": 0.46, "openai": 0.46, "time": 0.46}

REMAINING_ROUND_NUM = {
    "opening_for": 3,
    "opening_against": 2,
    "rebuttal_for": 1,
    "rebuttal_against": 0,
    "closing_for": 0,
    "closing_against": 0,
}
