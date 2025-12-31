from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import json
import os

from logic.suspect import Suspect

print("MAIN.PY LANCÉ")

app = FastAPI( title="Space Investigation AI",
    description="API des suspects",
    version="0.1.0",
    docs_url="/docs",      # explicitement
    redoc_url="/redoc",    # ReDoc
    openapi_url="/openapi.json")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Charger tous les suspects automatiquement
suspects = {}
suspect_folder = os.path.join(BASE_DIR, "data", "suspects")
for filename in os.listdir(suspect_folder):
    if filename.endswith(".json"):
        with open(os.path.join(suspect_folder, filename), encoding="utf-8") as f:
            data = json.load(f)
            s = Suspect(data)
            suspects[s.name] = s

# Requête POST
class AskRequest(BaseModel):
    suspect: str
    question: str

@app.post("/ask_suspect")
def ask_suspect(req: AskRequest):
    if req.suspect not in suspects:
        return {"error": "Suspect inconnu"}
    s = suspects[req.suspect]
    return {"text": s.answer(req.question), "emotion": s.state}

@app.get("/")
def root():
    return {"status": "ok"}


if __name__ == "__main__":
    print("DÉMARRAGE UVICORN")
    uvicorn.run(app, host="127.0.0.1", port=8000)