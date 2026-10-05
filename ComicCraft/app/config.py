from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
class Settings(BaseSettings):
    gemini_api_key:str=''; hf_api_key:str=''; gemini_model:str='gemini-2.5-flash'; gemini_story_model:str='gemini-2.5-flash'; image_provider:str='huggingface'; image_model:str='stabilityai/stable-diffusion-xl-base-1.0'; demo_mode:bool=True; panel_count:int=5
    model_config=SettingsConfigDict(env_file='.env',extra='ignore')
@lru_cache
def get_settings(): return Settings()
