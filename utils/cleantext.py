import re


def clean_text(text: str) -> str:
    # Replace multiple newlines/tabs with a single space
    text = re.sub(r"\s*\n\s*", " ", text)
    return text.strip()
