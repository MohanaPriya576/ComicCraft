from pydantic import BaseModel,Field
class PromptRequest(BaseModel):
    story_prompt:str=Field(...,min_length=5,max_length=2000)
    character_name:str=Field(...,min_length=1,max_length=100)
    setting:str=Field(...,min_length=1,max_length=100)
    tone:str=Field(...,min_length=1,max_length=50)
    art_style:str=Field(...,min_length=1,max_length=80)
