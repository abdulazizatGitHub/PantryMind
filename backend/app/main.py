from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import detect, ocr, pantry, recipes
from .config import Config


app = FastAPI(title='FoodWaste API')
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True)


app.include_router(detect.router, prefix='/api')
app.include_router(ocr.router, prefix='/api')
app.include_router(pantry.router, prefix='/api')
app.include_router(recipes.router, prefix='/api')


@app.get('/')
def root():
    return {'ok': True}


# run with: uvicorn app.main:app --reload --host 0.0.0.0 --port 8000