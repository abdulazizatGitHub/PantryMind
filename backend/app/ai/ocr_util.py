import easyocr
import re
import dateparser
from io import BytesIO
from PIL import Image
import numpy as np
from app.config import Config


reader = easyocr.Reader(['en'], gpu=Config.EASYOCR_GPU)


DATE_PATTERNS = [
    r"(\d{4}[-\/.]\d{1,2}[-\/.]\d{1,2})",
    r"(\d{1,2}[-\/.]\d{1,2}[-\/.]\d{2,4})",
    r"(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*\s*\d{1,2},?\s*\d{2,4}",
    r"BEST\s*BY[:\s]*([A-Z0-9\-/\s]+)"
]


def ocr_date_from_crop(image_bytes: bytes):
    img = Image.open(BytesIO(image_bytes)).convert('RGB')
    results = reader.readtext(np.array(img))
    text = " ".join([t[1] for t in results]).upper()
    for pat in DATE_PATTERNS:
        m = re.search(pat, text)
        if m:
            text_found = m.group(0)
            dt = dateparser.parse(text_found)
            if dt:
                return dt.date(), text
    return None, text