from fastapi import FastAPI, Form , HTTPException, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import instaloader
import os
import torch
from transformers import DistilBertForSequenceClassification,DistilBertTokenizer

app = FastAPI()
app.mount('/static',StaticFiles(directory="static"),name="static")

templates = Jinja2Templates(directory="templates")

model = DistilBertForSequenceClassification.from_pretrained('./saved_model')
tokenizer = DistilBertTokenizer.from_pretrained('./saved_tokenizer')

session_file_path = 'session.json'

class PredictionRequest(BaseModel):
    link: str

@app.get('/',response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html",{"request":request})

@app.post('/api/v1/analyze')
async def root(link: str = Form(...)):
    username = os.getenv("INSTAGRAM_USERNAME")
    password = os.getenv("INSTAGRAM_PASSWORD")
    L=instaloader.Instaloader()
    # Load session if it exists
    if os.path.exists(session_file_path):
        L.load_session_from_file(username=username)
    else:
        # Login and save session if it does not exist
        try:
            L.login(user="chatlakanhaiya", passwd="Helloworld@12")
            L.save_session_to_file()
        except Exception as e:
            print(e)
            raise HTTPException(status_code=500, detail="Login Failed")
    
    try:
        shortcode = link.split('/')[-2]
        post = instaloader.Post.from_shortcode(context=L.context,shortcode=shortcode)
    except Exception as e1:
        raise HTTPException(status_code=400,detail="Error Fetching post:"+str(e1))
    
    data = []
    comments = []
    for i, comment in enumerate(post.get_comments()):
        if i >= 100:
            break
        data.append({"username": comment.owner.username,"text":comment.text})
        
    encodings= tokenizer([item['text'] for item in data],truncation=True,padding=True,return_tensors='pt')
    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']

    with torch.no_grad():
        outputs= model(input_ids=input_ids,attention_mask=attention_mask)
        logits = outputs.logits
        preds= torch.sigmoid(logits) > 0.5
        preds = preds.int().tolist()

    classes = ['age', 'ethnicity', 'gender', 'not_cyberbullying', 'other_cyberbullying', 'religion']

    predictions = []
    notbully_count=0
    bully_count=0
    religion_count=0
    age_count=0
    ethnicity_count=0
    gender_count=0
    for comment,pred in zip(data,preds):
        predicted_class_names = [classes[i] for i, p in enumerate(pred) if p == 1]
        if "age" in predicted_class_names:
            age_count += 1
        if "religion" in predicted_class_names:
            religion_count += 1
        if "ethnicity" in predicted_class_names:
            ethnicity_count += 1
        if "gender" in predicted_class_names:
            gender_count += 1
        if "other_cyberbullying" in predicted_class_names:
            bully_count += 1
        if not any(c in predicted_class_names for c in ["age", "religion", "ethnicity", "gender", "other_cyberbullying"]):
            notbully_count += 1
        predictions.append({
            'user': comment['username'],
            'text': comment['text'],
            'sentiment':predicted_class_names})
    
    
    return {
        'predictions': predictions,
        'counts': {
            'total': len(predictions),
            'not_bullying': notbully_count,
            'bullying': bully_count,
            'age': age_count,
            'religion': religion_count,
            'ethnicity': ethnicity_count,
            'gender': gender_count
        }
    }


