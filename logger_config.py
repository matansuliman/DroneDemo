import logging

def setup_logger(name="app", log_file="app.log", level=logging.DEBUG):
    # Create a custom logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Handlers
    file_handler = logging.FileHandler(log_file, mode="w", delay=False)
    stream_handler = logging.StreamHandler()

    # Formatters
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # Add handlers (avoid duplicates if setup is called twice)
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger


def log_multiline(logger, msg: str):
    for line in msg.splitlines():
        logger.log(level=logger.getEffectiveLevel(),msg=line)
