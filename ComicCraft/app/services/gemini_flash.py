import json
from app.config import get_settings
def demo(req):
    names=[('The Call','discovers a mysterious clue'),('Into the Unknown','follows the clue'),('The Challenge','faces a surprising obstacle'),('The Turning Point','finds a clever solution'),('A New Beginning','makes a hopeful discovery')]
    return [{'panel_number':i+1,'title':t,'scene_description':f'{req.character_name} {d} in {req.setting}.','image_prompt':f'comic book panel, {req.art_style}, {req.tone} mood, {req.character_name}, {req.setting}, {d}, cinematic, expressive, detailed'} for i,(t,d) in enumerate(names)]
def generate_outline(req):
    s=get_settings()
    if s.demo_mode or not s.gemini_api_key:return demo(req)
    from google import genai
    c=genai.Client(api_key=s.gemini_api_key)
    p=f'''Create exactly {s.panel_count} coherent comic panels. Story: {req.story_prompt}. Character: {req.character_name}. Setting: {req.setting}. Tone: {req.tone}. Art style: {req.art_style}. Return ONLY JSON {{"panels":[{{"panel_number":1,"title":"","scene_description":"","image_prompt":""}}]}}.'''
    r=c.models.generate_content(model=s.gemini_model,contents=p,config={'response_mime_type':'application/json'})
    panels=json.loads(r.text)['panels']
    if len(panels)!=s.panel_count:raise ValueError('Unexpected panel count from Gemini')
    return panels
