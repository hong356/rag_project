from pathlib import Path
from pypdf import PdfReader
from logger import rag_logger

def load_document(file_path: str) -> str:
    """加载txt / md / pdf，返回完整文本"""
    path = Path(file_path)
    if not path.exists():
        rag_logger.error(f"文件不存在：{file_path}")
        raise FileNotFoundError(f"{file_path}不存在")

    suffix = path.suffix.lower()
    full_text = ""
    if suffix in [".txt", ".md"]:
        with open(path, "r", encoding="utf-8") as f:
            full_text = f.read()
    elif suffix == ".pdf":
        reader = PdfReader(path)
        for page in reader.pages:
            page_text = page.extract_text() or ""
            full_text += page_text
    else:
        rag_logger.warning(f"不支持的文件类型 {suffix}")
        raise ValueError("仅支持 txt md pdf")

    rag_logger.info(f"文档加载完成，总字符数：{len(full_text)}")
    return full_text