from motor.motor_asyncio import AsyncIOMotorClient
from app.config import Config


client = AsyncIOMotorClient(Config.MONGO_URI)
db = client[Config.MONGO_DB_NAME]


# Collections: users, pantry_items, events, recipes_cache
users_col = db['users']
pantry_col = db['pantry_items']
events_col = db['events']
recipes_col = db['recipes_cache']