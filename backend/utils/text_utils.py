import re

def clean_text(text: str) -> str:
    text = re.sub(r"[\x00-\x1F\x7F]", " ", text)
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<.*?>", "", text)
    return text.strip()

def chunk_text(text, max_tokens=400):
    sentences = text.split('. ')
    chunks, current_chunk, current_len = [], "", 0
    for sentence in sentences:
        current_chunk += sentence + '. '
        current_len += len(sentence.split())
        if current_len >= max_tokens:
            chunks.append(current_chunk.strip())
            current_chunk, current_len = "", 0
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks
