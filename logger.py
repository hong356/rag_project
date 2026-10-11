import logging
import os

if not os.path.exists("logs"):
    os.mkdir("logs")

def setup_logger(name: str):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # 避免重复添加handler
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    # 输出到文件
    file_handler = logging.FileHandler("logs/rag.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    # 输出控制台
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger

rag_logger = setup_logger("rag-system")