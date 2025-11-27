import logging
from logging import FileHandler, StreamHandler


def get_logger(name: str):
    logger = logging.getLogger(name)

    # すでにハンドラーが設定されている場合はスキップ
    if logger.hasHandlers():
        return logger

    # ロガーのレベル設定
    logger.setLevel(logging.INFO)

    # フォーマッターの作成
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # ファイルハンドラーの設定
    file_handler = FileHandler(f'./log/{name}.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # ストリームハンドラーの設定
    stream_handler = StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    # ハンドラーの追加
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger
