import json
from app.config import get_settings
def generate_story(outline,req):
    s=get_settings()
    if s.demo_mode or not s.gemini_api_key:
        return [{**p,'narration':f'{req.character_name} moves through {req.setting}, guided by a strange clue.','dialogue':f'{req.character_name}: "I will discover the truth!"','caption':f'{req.tone.title()} moment in {req.setting}.'} for p in outline]
    from google import genai
    c=genai.Client(api_key=s.gemini_api_key)
    p=f'''Expand this outline into exactly {len(outline)} comic panels. Character {req.character_name}, tone {req.tone}. Return ONLY JSON {{"panels":[{{"panel_number":1,"title":"","scene_description":"","image_prompt":"","narration":"","dialogue":"","caption":""}}]}}. Outline: {json.dumps(outline)}'''
    r=c.models.generate_content(model=s.gemini_story_model,contents=p,config={'response_mime_type':'application/json'})
    return json.loads(r.text)['panels']
