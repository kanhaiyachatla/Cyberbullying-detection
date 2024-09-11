from fastapi import FastAPI, Form , HTTPException, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import instaloader
import torch
from transformers import DistilBertForSequenceClassification,DistilBertTokenizer

app = FastAPI()
app.mount('/static',StaticFiles(directory="static"),name="static")
templates = Jinja2Templates(directory="templates")

model = DistilBertForSequenceClassification.from_pretrained('./saved_model')
tokenizer = DistilBertTokenizer.from_pretrained('./saved_tokenizer')


class PredictionRequest(BaseModel):
    link: str

@app.get('/',response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html",{"request":request})

@app.post('/api/v1/analyze')
async def root(link: str = Form(...)):
    L = instaloader.Instaloader()
    try:
        L = instaloader.Instaloader()
        L.load_session_from_file('filename') #replace with any file name
# If session doesn't exist, create it
        if not L.context.is_logged_in:
            L.login('chatlakanhaiya', 'Kanhaiya@12') # Here replace with your Instagram Credentials
            L.save_session_to_file()
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500,detail="Login Failed")
    
    try:
        shortcode = link.split('/')[-2]
        post = instaloader.Post.from_shortcode(context=L.context,shortcode=shortcode)
    except Exception as e1:
        raise HTTPException(status_code=400,detail="Error Fetching post:"+str(e1))
    
    data = []
    comments = []
    for i, comment in enumerate(post.get_comments()):
        if i >= 200:
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
    for comment,pred in zip(data,preds):
        predicted_class_names = [classes[i] for i, p in enumerate(pred) if p == 1]
        predictions.append({
            'user': comment['username'],
            'text': comment['text'],
            'sentiment':predicted_class_names})
    return predictions


