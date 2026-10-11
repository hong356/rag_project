from logger import rag_logger
from typing import List

def split_text(text: str, chunk_size: int = 512, overlap: int = 80) -> List[str]:
    """滑动窗口切分文本，chunk_size块大小，overlap重叠字符"""
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        # 滑动，保留重叠部分
        start = end - overlap
    rag_logger.info(f"文本切分完成，一共 {len(chunks)} 个片段")
    return chunks