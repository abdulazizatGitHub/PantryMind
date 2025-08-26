from fastapi import APIRouter
from ..models import PantryItemIn
from ..db import pantry_col
from datetime import datetime


router = APIRouter()


@router.post('/pantry/upsert')
async def upsert(item: PantryItemIn):
    doc = item.dict()
    doc['created_at'] = datetime.utcnow()
    res = await pantry_col.insert_one(doc)
    return { 'inserted_id': str(res.inserted_id) }


@router.get('/pantry/list')
async def list_items():
    docs = []
    async for d in pantry_col.find().sort('created_at', -1).limit(200):
        d['_id'] = str(d['_id'])
        docs.append(d)
    return docs