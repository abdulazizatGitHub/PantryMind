from fastapi import APIRouter, UploadFile, File
from ..ai.yolov8_infer import run_yolo_on_image_bytes
from ..ai.ocr_util import ocr_date_from_crop
from ..db import pantry_col
import uuid, base64


router = APIRouter()


@router.post('/detect')
async def detect(image: UploadFile = File(...)):
    img_bytes = await image.read()
    raw = run_yolo_on_image_bytes(img_bytes)
    # prepare crops for OCR and store minimal thumbnail in DB as base64
    results = []
    for r in raw:
        # For MVP we will not crop images on server to avoid extra image libs; return bounding boxes and names
        results.append({
            'name': r['name'],
            'bbox': r['bbox'],
            'conf': r['conf']
        })
    return {'items': results}