import random

class Suspect:
    def __init__(self, data: dict):
        self.name = data["name"]
        self.guilty = data["guilty"]
        self.personality = data["personality"]

        # État dynamique
        self.state = {
            "fear": self.personality["fear"],
            "confidence": self.personality["confidence"],
            "pressure": 0.0
        }

        # Mémoire des questions/réponses
        self.memory = []

        self.official_story = data["official_story"]

    def update_state(self, question: str):
        if "où" in question.lower():
            self.state["pressure"] += 0.1
        if "pourquoi" in question.lower():
            self.state["pressure"] += 0.2

        if self.guilty:
            self.state["fear"] += 0.1 * self.state["pressure"]

        # Clamp entre 0 et 1
        for k in self.state:
            self.state[k] = max(0.0, min(1.0, self.state[k]))

    def choose_strategy(self):
        if not self.guilty:
            return "truth"
        if self.state["fear"] < 0.65:
            return "lie_confident"
        
        if self.state["fear"] < 0.72:
            return "lie_vague"
        return "deflect"

    def answer(self, question: str) -> str:
        self.update_state(question)
        strategy = self.choose_strategy()

        # Vérifie la mémoire pour éviter de répéter exactement la même réponse
        for q, a in self.memory:
            if question.lower() == q.lower():
                return f"Comme je l'ai dit : {a}"

        # Générer la réponse selon la stratégie
        if strategy == "truth":
            response = self.official_story["alibi"][0] if self.official_story["alibi"] else "Je n'ai rien à cacher."
        elif strategy == "lie_confident":
            response = "Je vous ai déjà dit exactement ce que j'ai fait."
        elif strategy == "lie_vague":
            response = "Je… c'était une soirée normale, rien de spécial."
        elif strategy == "deflect":
            response = "Pourquoi vous insistez autant ?"
        else:
            response = "..."

        # Ajouter dans la mémoire
        self.memory.append((question, response))

        return response