LOG_PROGRESS = True

EXIT_FAILURE = -1
EXIT_SUCCESS = 0

DEFAULT_NUM_PAGES = 50
DEFAULT_NUM_THREADS = 5 # Set to 1 for no threading

DEFAULT_DB_NAME = "chatbot"
DEFAULT_DB_PATH = "chroma_data"

USE_THREADING = True
LOG_TIME_INFO = True

TOTAL_NUM_PAGES: int = 5000

ROUNDS: int = 500
BATCH_LEN: int = int(TOTAL_NUM_PAGES / ROUNDS)

BATCH_SIZE = 256

# NN related constants

EMBED_DIM = 512
HIDDEN_DIM = 64

NUM_EPOCHS = 10

MAX_DECODER_OUTPUT_LENGTH = 20

TRAIN_UPDATE_MSG = "Epoch {n}: Train loss={t_loss:.2f} | Eval loss = {e_loss:.2f}"
