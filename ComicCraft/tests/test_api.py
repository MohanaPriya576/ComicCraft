import os
os.environ['DEMO_MODE']='true'
from fastapi.testclient import TestClient
from app.main import app
c=TestClient(app)
def data():return {'story_prompt':'A brave fox in a magical forest','character_name':'Luna','setting':'Forest','tone':'Funny','art_style':'Comic Book'}
def test_home():assert c.get('/').status_code==200
def test_docs():assert c.get('/docs').status_code==200
def test_json():
 r=c.post('/generate-comic/json',json=data()); assert r.status_code==200; assert len(r.json()['panels'])==5; assert r.json()['pdf_url'].endswith('.pdf')
def test_form():assert c.post('/generate',data=data()).status_code==200
def test_image():assert c.post('/test-image',data={'prompt':'fox'}).json()['success'] is True
def test_validation():
 d=data();d['story_prompt']='x';assert c.post('/generate-comic/json',json=d).status_code==422
