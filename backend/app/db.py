from motor.motor_asyncio import AsyncIOMotorClient
from .config import settings


client = AsyncIOMotorClient(settings.MONGODB_URI)
db = client[settings.MONGO_DB_NAME]


# Collections: users, pantry_items, events, recipes_cache
users_col = db['users']
pantry_col = db['pantry_items']
events_col = db['events']
recipes_col = db['recipes_cache']