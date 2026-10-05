from fastapi import APIRouter,Request,Form,HTTPException
from fastapi.templating import Jinja2Templates
from app.schemas import PromptRequest
from app.services.gemini_flash import generate_outline
from app.services.gemini_pro import generate_story
from app.services.image_generator import generate_image
from app.services.layout_builder import build_comic_layout
from app.services.exporters import save_pdf
from pathlib import Path
router=APIRouter(); templates=Jinja2Templates(directory=str(Path(__file__).resolve().parents[1]/'templates'))
def run(req):
    panels=generate_story(generate_outline(req),req)
    for p in panels:p['image_path']=generate_image(p['image_prompt'],p['panel_number'])
    layout=build_comic_layout(panels); return layout,save_pdf(layout)
@router.get('/')
async def home(request:Request):return templates.TemplateResponse('index.html',{'request':request})
@router.post('/generate')
async def generate(request:Request,story_prompt:str=Form(...),character_name:str=Form(...),setting:str=Form(...),tone:str=Form(...),art_style:str=Form(...)):
    try:
        layout,pdf=run(PromptRequest(story_prompt=story_prompt,character_name=character_name,setting=setting,tone=tone,art_style=art_style)); return templates.TemplateResponse('comic_preview.html',{'request':request,'layout':layout,'pdf_url':pdf})
    except Exception as e: raise HTTPException(500,str(e))
@router.post('/generate-comic/json')
async def generate_json(req:PromptRequest):
    try:
        layout,pdf=run(req); return {'title':f'{req.character_name} Comic','panels':layout,'pdf_url':pdf}
    except Exception as e: raise HTTPException(500,str(e))
@router.post('/test-image')
async def test_image(prompt:str=Form(...)): return {'success':True,'image_path':generate_image(prompt,0)}
@router.get('/export-success')
async def export_success(request:Request):return templates.TemplateResponse('export_success.html',{'request':request})
