import logging
import os

from logging.handlers import RotatingFileHandler


# =========================================================
# LOG DIRECTORY
# =========================================================

LOG_DIR = "logs"

os.makedirs(LOG_DIR, exist_ok=True)


# =========================================================
# LOGGER CREATOR
# =========================================================

def create_logger(
    logger_name: str,
    log_file: str,
    level=logging.INFO
):

    logger = logging.getLogger(logger_name)

    logger.setLevel(level)

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    # -----------------------------------------------------
    # FILE HANDLER
    # -----------------------------------------------------

    file_handler = RotatingFileHandler(
        filename=f"{LOG_DIR}/{log_file}",
        maxBytes=10 * 1024 * 1024,
        backupCount=5
    )

    file_handler.setFormatter(formatter)

    # -----------------------------------------------------
    # CONSOLE HANDLER
    # -----------------------------------------------------

    console_handler = logging.StreamHandler()

    console_handler.setFormatter(formatter)

    # -----------------------------------------------------
    # ADD HANDLERS
    # -----------------------------------------------------

    logger.addHandler(file_handler)

    logger.addHandler(console_handler)

    logger.propagate = False

    return logger


# =========================================================
# APPLICATION LOGGER
# =========================================================

app_logger = create_logger(
    logger_name="app_logger",
    log_file="app.log"
)

# =========================================================
# ERROR LOGGER
# =========================================================

error_logger = create_logger(
    logger_name="error_logger",
    log_file="error.log"
)

# =========================================================
# VIOLATION LOGGER
# =========================================================

violation_logger = create_logger(
    logger_name="violation_logger",
    log_file="violations.log"
)

# =========================================================
# PERFORMANCE LOGGER
# =========================================================

performance_logger = create_logger(
    logger_name="performance_logger",
    log_file="performance.log"
)
