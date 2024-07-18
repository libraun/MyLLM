# Exit codes
EXIT_FAILURE = -1
EXIT_SUCCESS = 0

# Chroma db & threading constants
DEFAULT_DB_NAME = "chatbot"
DEFAULT_DB_PATH = "chroma_data"

LOG_PROGRESS = True

USE_THREADING = True
LOG_TIME_INFO = True

DEFAULT_NUM_PAGES = 50
DEFAULT_NUM_THREADS = 5 # Set to 1 for no threading

TOTAL_NUM_PAGES: int = 5000

ROUNDS: int = 500 # The number of times worker threads will work... on threads
BATCH_LEN: int = int(TOTAL_NUM_PAGES / ROUNDS) # The number of entries per worker thread

# NN constants
DEFAULT_JSON_DATASET = "./Topical-Chat/conversations/train.json"

HIDDEN_DIM = 64
BATCH_SIZE = 64
SPECIAL_TOKENS = [
    "<PAD_IDX>", "<UNK_IDX>",
    "<BOS_IDX>", "<EOS_IDX>", 
    "<BEGIN_MD_IDX>", "<END_MD_IDX>"
]