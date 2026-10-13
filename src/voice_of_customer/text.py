"""Conservative Arabic normalization and local personal-data masking."""

import re
import unicodedata


def mask_pii(text: str) -> str:
    text = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "[EMAIL]", text)
    return re.sub(r"(?<!\w)(?:\+?\d[\d ()-]{7,}\d)(?!\w)", "[PHONE]", text)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", mask_pii(text)).lower()
    text = re.sub(r"[\u064b-\u065f\u0670\u0640]", "", text)
    text = re.sub("[أإآٱ]", "ا", text).replace("ى", "ي")
    return re.sub(r"\s+", " ", text).strip()
