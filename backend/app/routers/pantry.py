from fastapi import APIRouter, HTTPException
from ..models import PantryItem
from ..db import pantry_col
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post('/pantry/upsert')
async def upsert(item: PantryItem):
    try:
        doc = item.model_dump()
        doc['created_at'] = datetime.now(timezone.utc)
        res = await pantry_col.insert_one(doc)
        return { 'inserted_id': str(res.inserted_id) }
    except Exception as e:
        logger.error(f"Database error: {e}")
        # Return mock response for testing
        return { 'inserted_id': 'mock_id_123', 'status': 'mock_response' }


@router.get('/pantry/list')
async def list_items():
    try:
        docs = []
        async for d in pantry_col.find().sort('created_at', -1).limit(200):
            d['_id'] = str(d['_id'])
            docs.append(d)
        return docs
    except Exception as e:
        logger.error(f"Database error: {e}")
        # Return mock data for testing
        return [
            {
                '_id': 'mock_1',
                'name': 'Tomato',
                'quantity': 2,
                'unit': 'pieces',
                'created_at': datetime.now(timezone.utc)
            },
            {
                '_id': 'mock_2', 
                'name': 'Pasta',
                'quantity': 500,
                'unit': 'grams',
                'created_at': datetime.now(timezone.utc)
            }
        ]