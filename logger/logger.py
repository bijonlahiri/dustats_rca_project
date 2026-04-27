import os
import logging
from datetime import datetime

LOG_FILE = f"log_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log"
LOG_DIR = "logs"
LOG_FILEPATH = os.path.join(LOG_DIR, LOG_FILE)

os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=LOG_FILEPATH,
    format="[%(asctime)s] - %(levelname)s - %(filename)s - %(lineno)s - %(message)s",
    level=logging.INFO
)