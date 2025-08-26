from pydantic import BaseSettings


class Settings(BaseSettings):
    MONGODB_URI: str = "mongodb://localhost:27017"
    MONGO_DB_NAME: str = "foodwaste"
    OPENAI_API_KEY: str | None = None
    RECIPE_INDEX_PATH: str = "../data/recipe_index.faiss"
    RECIPE_META_PATH: str = "../data/recipes_meta.parquet"
    YOLO_MODEL_PATH: str = "../models/yolov8n.pt" # default; download from ultralytics if needed
    ALLOWED_HOSTS: list[str] = ["*"]


class Config:
    env_file = "../.env"


settings = Settings()