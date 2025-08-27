from datetime import datetime, date
from typing import List, Optional, Dict, Any
from bson import ObjectId

class PantryItem:
    def __init__(self, name: str, quantity: float = 1.0, unit: str = "unit", 
                 expiry_date: Optional[date] = None, image_url: Optional[str] = None,
                 confidence: Optional[float] = None, category: Optional[str] = None,
                 created_at: Optional[datetime] = None, updated_at: Optional[datetime] = None):
        self.name = name
        self.quantity = quantity
        self.unit = unit
        self.expiry_date = expiry_date
        self.image_url = image_url
        self.confidence = confidence
        self.category = category
        self.created_at = created_at or datetime.utcnow()
        self.updated_at = updated_at or datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'quantity': self.quantity,
            'unit': self.unit,
            'expiry_date': self.expiry_date.isoformat() if self.expiry_date else None,
            'image_url': self.image_url,
            'confidence': self.confidence,
            'category': self.category,
            'created_at': self.created_at,
            'updated_at': self.updated_at
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

class Recipe:
    def __init__(self, title: str, ingredients: List[str], instructions: List[str],
                 cooking_time: int, difficulty: str = "medium", cuisine: str = "general",
                 tags: Optional[List[str]] = None, image_url: Optional[str] = None,
                 created_at: Optional[datetime] = None):
        self.title = title
        self.ingredients = ingredients
        self.instructions = instructions
        self.cooking_time = cooking_time
        self.difficulty = difficulty
        self.cuisine = cuisine
        self.tags = tags or []
        self.image_url = image_url
        self.created_at = created_at or datetime.utcnow()
    
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
            'created_at': self.created_at
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

class UserInteraction:
    def __init__(self, action: str, item_id: Optional[str] = None, recipe_id: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None, created_at: Optional[datetime] = None):
        self.action = action  # 'scan_item', 'add_item', 'view_recipe', 'waste_reduced'
        self.item_id = item_id
        self.recipe_id = recipe_id
        self.metadata = metadata or {}
        self.created_at = created_at or datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'action': self.action,
            'item_id': self.item_id,
            'recipe_id': self.recipe_id,
            'metadata': self.metadata,
            'created_at': self.created_at
        }