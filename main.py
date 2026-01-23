from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import json
import os

from logic.suspect import Suspect
from logic.logger import log_interaction
from backend.llm import ask_llm
PRESSURE_BREAK = 0.95
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

# Charger tous les indices automatiquement
indices = {}
indices_folder = os.path.join(BASE_DIR, "data", "indices")

for filename in os.listdir(indices_folder):
    if filename.endswith(".json"):
        with open(os.path.join(indices_folder, filename), encoding="utf-8") as f:
            data = json.load(f)
            indices[data["id"]] = data

# Requête POST
class AskRequest(BaseModel):
    suspect: str
    question: str

def detect_evidence(question: str, indices: dict) -> list:
    question = question.lower()
    triggered = []

    for evidence in indices.values():
        for kw in evidence["keywords"]:
            if kw in question:
                triggered.append(evidence)
                break

    return triggered

@app.post("/ask_suspect")

def ask_suspect(req: AskRequest):
    if req.suspect not in suspects:
        return {"error": "Suspect inconnu"}
    s = suspects[req.suspect]
    s.update_state(req.question)
    triggered = detect_evidence(req.question, indices)

    for evidence in triggered:
        bonus = evidence["effects"].get(s.name, 0.0)

        if bonus > 0:
            s.state["pressure"] = min(1.0, s.state["pressure"] + bonus)

            if s.state["pressure"] > 0.6:
                s.strategy = "deflect"
    if s.state["pressure"] >= PRESSURE_BREAK:
     tempanswer = "Je n’ai plus rien à vous dire."
    else:
        tempanswer = ask_llm(
        s,req.question
        )
    log_interaction(
        suspect=s.name,
        question=req.question,
        strategy=s.strategy,
        state=s.state,
        answer=tempanswer
    )
    s.memory.append({
    "question": req.question,
    "answer": tempanswer,
    "strategy": s.strategy,
    "state": s.state.copy(),
})
    return {"text":tempanswer, "emotion": s.state}
    

@app.get("/")
def root():
    return {"status": "ok"}


if __name__ == "__main__":
    print("DÉMARRAGE UVICORN")
    uvicorn.run(app, host="127.0.0.1", port=8000)