from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from app.routes import router
ROOT=Path(__file__).resolve().parents[1]
app=FastAPI(title='ComicCraft - AI Comic Story Creator',version='1.0.0',description='AI comic generation using Gemini and Stable Diffusion-compatible image generation.')
app.mount('/static',StaticFiles(directory=str(ROOT/'static')),name='static'); app.include_router(router)
