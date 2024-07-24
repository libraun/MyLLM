# Exit Codes
EXIT_FAILURE: int = -1
EXIT_SUCCESS: int = 0

# Chroma related constants (building db)

DEFAULT_NUM_PAGES: int = 50
DEFAULT_NUM_THREADS: int = 5 # Set to 1 for no threading

DEFAULT_DB_NAME: str = "chatbot"
DEFAULT_DB_PATH: str = "chroma_data"

USE_THREADING: bool = True
LOG_TIME_INFO: bool = True

TOTAL_NUM_PAGES: int = 5000
COLLECTION_ROUNDS: int = 500
CHROMA_BATCH_LEN: int = int(TOTAL_NUM_PAGES / COLLECTION_ROUNDS)

# Preprocessing Constants
TOPICALCHAT_PATH: str = "./Topical-Chat/conversations/train.json"

DEFAULT_MAX_MSGS: int = 50000
STRIP_INPUT_STOPWORDS: bool = True 

''' NN related constants '''

REVERSE_ENCODER_INPUTS: bool = False

# Size ratio for training and validation splits. A value of 0.8 means 80% of the data is used for training.
DATASET_SPLIT_RATIO: float = 0.8

# Tokens used to indicate certain text spans.
SPECIAL_TOKENS: list = [
    "<PAD>", "<UNK>",
    "<BOS>", "<EOS>", 
    "<BMD>", "<EMD>",
]

HIDDEN_DIM: int = 64
NUM_EPOCHS: int = 10

MAX_DECODER_OUTPUT_LENGTH: int = 20

TRAIN_UPDATE_MSG: str = "Epoch {n}: Train loss={t_loss:.2f} | Eval loss = {e_loss:.2f}"
