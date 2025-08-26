from fastapi import APIRouter, UploadFile, File
from ..ai.ocr_util import ocr_date_from_crop


router = APIRouter()


@router.post('/ocr')
async def ocr_endpoint(image: UploadFile = File(...)):
    b = await image.read()
    dt, text = ocr_date_from_crop(b)
    return { 'expiry_date': str(dt) if dt else None, 'text': text }