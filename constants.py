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
BATCH_LEN: int = int(TOTAL_NUM_PAGES / COLLECTION_ROUNDS)

LOG_COLLECTION_PROGRESS = True # Whether or not to log number of documents processed by workers

# Preprocessing Constants
TOPICALCHAT_PATH: str = "./Topical-Chat/conversations/train.json"

DEFAULT_MAX_MSGS: int = 50000
STRIP_INPUT_STOPWORDS: bool = False 

# NN related constants

DATALOADER_BATCH_SIZE: int = 256
SHUFFLE_DATALOADERS: bool = False

SPECIAL_TOKENS: list = [
    "<UNK_IDX>", "<PAD_IDX>",
    "<BOS_IDX>", "<EOS_IDX>", 
    "<BEGIN_MD_IDX>", "<END_MD_IDX>"
]

HIDDEN_DIM: int = 64
NUM_EPOCHS: int = 10

MAX_DECODER_OUTPUT_LENGTH: int = 20

TRAIN_UPDATE_MSG: str = "Epoch {n}: Train loss={t_loss:.2f} | Eval loss = {e_loss:.2f}"
