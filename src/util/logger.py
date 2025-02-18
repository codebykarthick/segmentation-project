from datetime import datetime
import logging
import os

LOG_PATH = "logs"
log_file = os.path.join(
    LOG_PATH, f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
os.makedirs(LOG_PATH, exist_ok=True)


def setup_logger():
    """ Setup a logger for the model training. """
    # Logger
    logger = logging.getLogger("AssignmentLogger")
    logger.setLevel(logging.INFO)

    # Formatting
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # File handler
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
