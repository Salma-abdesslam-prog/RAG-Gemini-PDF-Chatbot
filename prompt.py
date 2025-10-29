import re

def process_user_prompt(prompt_text, chunk_size=200, overlap=50, lower=True, apply_corrections=False):
    """
    Clean, normalize, and chunk user prompt text for embedding or processing.
    
    Args:
        prompt_text (str): The raw user input text.
        chunk_size (int): Number of words per chunk.
        overlap (int): Number of overlapping words between chunks.
        lower (bool): Convert to lowercase for normalization.
        apply_corrections (bool): Should apply OCR correction (usually False for keyboard input).

    Returns:
        list[str]: List of cleaned text chunks.
    """

    # --- 1️⃣ Basic cleaning ---
    text = prompt_text.encode("utf-8", errors="ignore").decode("utf-8")
    text = re.sub(r'\s+', ' ', text).strip()

    # --- 2️⃣ Lowercase normalization ---
    if lower:
        text = text.lower()

    # --- 3️⃣ OCR corrections (only if needed) ---
    if apply_corrections:
        corrections = {
            '0': 'o',
            '1': 'l',
            '|': 'l',
            '“': '"',
            '”': '"',
            '‘': "'",
            '’': "'"
        }
        for wrong, right in corrections.items():
            text = text.replace(wrong, right)

    # --- 4️⃣ Chunking ---
    words = text.split()
    chunks = [
        " ".join(words[i:i + chunk_size])
        for i in range(0, len(words), chunk_size - overlap)
    ]

    return chunks
