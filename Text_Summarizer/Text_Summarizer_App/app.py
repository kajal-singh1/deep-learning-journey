from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# initialize our fastapi app
app = FastAPI(title="Text Summarizer App", description="Text Summarization using T5", version="1.0")

# model & tokenizer
model = T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# device
import torch
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

model.to(device)

# templates
templates = Jinja2Templates(directory="templates")

# Input schema for dialogue => string 
class DialougueInput(BaseModel):
    dialogue : str

# clean function
def clean_data(text):
    text = re.sub(r"\r\n", " ", text) # Lines
    text = re.sub(r"\s+", " ", text) # spaces
    text = re.sub(r"<.*?>", " ", text) # html tags
    text = text.strip().lower()
    return text

# Summarization Function
def summarize_dialougue(dialogue:str) -> str:
  dialogue = clean_data(dialogue)

  inputs = tokenizer(
      dialogue,
      padding="max_length",
      max_length=512,
      truncation=True,
      return_tensors="pt"
  ).to(device)

  model.to(device)
  targets = model.generate(
      input_ids = inputs["input_ids"],
      attention_mask = inputs["attention_mask"],
      max_length = 150,
      num_beams = 4,
      early_stopping = True
  )

  summary = tokenizer.decode(targets[0], skip_special_tokens=True)
  return summary

# API Endpoints
@app.post("/summarize/")
async def summarize(dialogue_input: DialougueInput):
    summary = summarize_dialougue(dialogue_input.dialogue)
    return {"summary": summary}

@app.get("/", response_class=HTMLResponse)
async def home(request : Request):
    return templates.TemplateResponse("index.html", {"request":request})

