import logging
from logging import FileHandler, StreamHandler


def get_logger(name: str):
    logger = logging.getLogger(name)

    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    file_handler = FileHandler(f'./log/{name}.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger
