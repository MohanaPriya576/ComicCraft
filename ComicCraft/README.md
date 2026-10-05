# ComicCraft - AI Comic Story Creator using Gemini Models

This implementation follows the supplied project document: FastAPI + Jinja2 frontend, Gemini for structured comic outline/story generation, Stable Diffusion-compatible image generation, five panels, and FPDF PDF export.

## Fast setup (Windows PowerShell)
```powershell
cd ComicCraft
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --reload
```
Open http://127.0.0.1:8000 and API docs at http://127.0.0.1:8000/docs.

### First test without API keys
Keep `DEMO_MODE=true` in `.env`. The application will generate deterministic demo illustrations and a real PDF, allowing you to verify the entire pipeline.

### Real Gemini + image generation
Set `DEMO_MODE=false`, add `GEMINI_API_KEY`, and set `IMAGE_PROVIDER=huggingface` plus `HF_API_KEY`. The image model is Stable Diffusion XL by default. For local Diffusers, set `IMAGE_PROVIDER=diffusers`; this requires substantial model download/storage and is much better with a compatible GPU.

## Test
```powershell
python -m pytest -q
```

## Project flow
1. User enters story prompt, character, setting, tone and art style.
2. Gemini creates a structured 5-panel outline.
3. Gemini expands each panel with narration/dialogue.
4. Stable Diffusion-compatible image generation creates one illustration per panel.
5. Layout builder combines text and images.
6. FPDF2 exports the comic to `static/exports`.
