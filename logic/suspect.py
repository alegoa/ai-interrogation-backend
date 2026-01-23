import random

class Suspect:
    def __init__(self, data: dict):
        self.name = data["name"]
        self.guilty = data["guilty"]
        self.alibi = data["alibi"]
        self.state = data["state"]
        self.identity = data.get("identity", {})
       
     
        if not self.guilty:
            self.strategy = "truth"
        else:
            self.strategy = "lie_confident"
        # État dynamique
        self.state = {
            "fear": self.state["fear"],
            "pressure": self.state["pressure"]
        }

        # Mémoire des questions/réponses
        self.memory = []

       

    def update_state(self, question: str):
        if self.is_repeated_question(question):
            self.state["pressure"] = min(1.0, self.state["pressure"] + 0.15)

            # Le suspect devient plus défensif
            if self.state["pressure"] > 0.6:
                self.strategy = "deflect"
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
        

        return response
    
    def is_repeated_question(self, question: str) -> bool:
        question = question.lower().strip()

        for past in self.memory:
            if question in past["question"].lower():
                return True

        return False