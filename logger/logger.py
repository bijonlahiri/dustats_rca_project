import os
import logging
from datetime import datetime

LOG_DIR = "/tmp/logs"
LOG_FILE = f"log_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log"
LOG_FILEPATH = os.path.join(LOG_DIR, LOG_FILE)

LOG_FORMAT = "[%(asctime)s] - %(levelname)s - %(filename)s - %(lineno)s - %(message)s"

logger = logging.getLogger()
logger.setLevel(logging.INFO)

if not logger.handlers:
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(stream_handler)

if not os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
    os.makedirs(LOG_DIR, exist_ok=True)
    file_handler = logging.FileHandler(LOG_FILEPATH)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(file_handler)