import logging
import sys
import os
from pythonjsonlogger import jsonlogger
import config

def setup_logging(log_level=logging.INFO, enable_file_logging=False):
    """Configures structured JSON logging."""
    logger = logging.getLogger()
    logger.setLevel(log_level)

    log_handler = logging.StreamHandler(sys.stdout)
    formatter = jsonlogger.JsonFormatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s %(filename)s %(lineno)d",
        datefmt="%Y-%m-%dT%H:%M:%S%z"
    )
    log_handler.setFormatter(formatter)
    logger.handlers.clear()
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
         logger.addHandler(log_handler)

    if enable_file_logging and config.LOG_DIR:
        log_file = os.path.join(config.LOG_DIR, "app.log")
        os.makedirs(config.LOG_DIR, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.handlers.clear()
        if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
             logger.addHandler(file_handler)

    return logger