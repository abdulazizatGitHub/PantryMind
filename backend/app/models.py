from pydantic import BaseModel, Field
from typing import Optional, List, Any
from datetime import date


class PantryItemIn(BaseModel):
    name: str
    quantity: Optional[float] = 1
    unit: Optional[str] = "unit"
    expiry_date: Optional[str] = None
    image_uri: Optional[str] = None
    confidence: Optional[float] = None


class PantryItemDB(PantryItemIn):
    id: Optional[str] = None


class RecipeOut(BaseModel):
    title: str
    ingredients: List[str]
    steps: List[str]
    uses_expiring: List[str]
    substitutions: List[Any] = []
    score: Optional[float] = 0.0


class DetectResult(BaseModel):
    name: str
    bbox: List[int]
    conf: float
    crop_uri: Optional[str] = None