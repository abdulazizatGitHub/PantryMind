from datetime import datetime, date, timezone
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from bson import ObjectId

class PantryItem(BaseModel):
    name: str
    quantity: float = 1.0
    unit: str = "unit"
    expiry_date: Optional[date] = None
    image_url: Optional[str] = None
    confidence: Optional[float] = None
    category: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'quantity': self.quantity,
            'unit': self.unit,
            'expiry_date': self.expiry_date.isoformat() if self.expiry_date else None,
            'image_url': self.image_url,
            'confidence': self.confidence,
            'category': self.category,
            'created_at': self.created_at or datetime.now(timezone.utc),
            'updated_at': self.updated_at or datetime.now(timezone.utc)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PantryItem':
        expiry_date = None
        if data.get('expiry_date'):
            try:
                expiry_date = datetime.fromisoformat(data['expiry_date']).date()
            except:
                pass
        
        return cls(
            name=data['name'],
            quantity=data.get('quantity', 1.0),
            unit=data.get('unit', 'unit'),
            expiry_date=expiry_date,
            image_url=data.get('image_url'),
            confidence=data.get('confidence'),
            category=data.get('category'),
            created_at=data.get('created_at'),
            updated_at=data.get('updated_at')
        )

class Recipe(BaseModel):
    title: str
    ingredients: List[str]
    instructions: List[str]
    cooking_time: int
    difficulty: str = "medium"
    cuisine: str = "general"
    tags: Optional[List[str]] = []
    image_url: Optional[str] = None
    created_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'title': self.title,
            'ingredients': self.ingredients,
            'instructions': self.instructions,
            'cooking_time': self.cooking_time,
            'difficulty': self.difficulty,
            'cuisine': self.cuisine,
            'tags': self.tags,
            'image_url': self.image_url,
            'created_at': self.created_at or datetime.now(timezone.utc)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Recipe':
        return cls(
            title=data['title'],
            ingredients=data['ingredients'],
            instructions=data['instructions'],
            cooking_time=data.get('cooking_time', 30),
            difficulty=data.get('difficulty', 'medium'),
            cuisine=data.get('cuisine', 'general'),
            tags=data.get('tags', []),
            image_url=data.get('image_url'),
            created_at=data.get('created_at')
        )

class UserInteraction(BaseModel):
    action: str  # 'scan_item', 'add_item', 'view_recipe', 'waste_reduced'
    item_id: Optional[str] = None
    recipe_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = {}
    created_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'action': self.action,
            'item_id': self.item_id,
            'recipe_id': self.recipe_id,
            'metadata': self.metadata,
            'created_at': self.created_at or datetime.now(timezone.utc)
        }