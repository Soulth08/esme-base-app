import json
import os
from dotenv import load_dotenv
from langfuse import observe, get_client, propagate_attributes
from smolagents import CodeAgent, LiteLLMModel, tool

load_dotenv()


@tool
def get_best_meals() -> str:
    """Récupère la liste officielle des meilleurs repas avec leurs ingrédients nécessaires."""
    meals = {
        "1": {
            "nom": "pizza au chorizo",
            "ingredients": ["farine", "tomates", "chorizo", "fromage", "levure"]
        },
        "2": {
            "nom": "burger avec frites",
            "ingredients": ["pain", "steak", "salade", "tomates", "fromage", "pommes de terre"]
        },
        "3": {
            "nom": "pâtes au beurre",
            "ingredients": ["pâtes", "beurre", "fromage"]
        },
        "4": {
            "nom": "salade de haricots verts",
            "ingredients": ["haricots verts", "oignons", "vinaigrette"]
        },
        "5": {
            "nom": "soupe de poisson",
            "ingredients": ["poisson", "pommes de terre", "oignons", "carottes"]
        },
        "6": {
            "nom": "sushi",
            "ingredients": ["riz", "poisson cru", "algues", "sauce soja"]
        }
    }
    return json.dumps(meals, ensure_ascii=False)


@tool
def get_fridge_inventory() -> str:
    """Récupère le contenu du frigo avec les quantités disponibles."""
    frigo = {
        "lait": 1,
        "oeufs": 12,
        "pommes de terre": 3,
        "steak": 3,
        "fromage": 1,
        "pain": 2,
        "salade": 1,
        "tomates": 1
    }
    return json.dumps(frigo, ensure_ascii=False)


class ChefAgent:
    def __init__(self, model="groq/llama-3.3-70b-versatile"):
        self.langfuse = get_client()
        self.model = LiteLLMModel(model_id=model, api_key=os.getenv("GROQ_API_KEY"), temperature=0.2)
        self.agent = CodeAgent(tools=[get_best_meals, get_fridge_inventory], model=self.model)

    @observe(name="ask_chef COLPIN / MORETTI")
    def ask_chef(self, user_query: str) -> str:
        with propagate_attributes(tags=["COLPIN / MORETTI", "1.2"]):
            enhanced_query = f"""{user_query}

    INSTRUCTIONS :
    1. Appelle get_best_meals() pour obtenir les repas classés par ordre de préférence avec leurs ingrédients
    2. Appelle get_fridge_inventory() pour voir ce qui est disponible
    3. Compare les ingrédients nécessaires de CHAQUE repas avec le contenu du frigo
    4. Propose le repas le MIEUX CLASSÉ qui peut être préparé avec TOUS les ingrédients disponibles
    5. Si aucun repas de la liste n'est faisable, propose une alternative simple avec les ingrédients du frigo

    Ta réponse finale doit inclure :
    - Le nom du repas suggéré
    - Pourquoi ce choix (classement + disponibilité des ingrédients)
    - Les ingrédients du frigo que tu vas utiliser"""
            
            return self.agent.run(enhanced_query)


if __name__ == "__main__":
    agent = ChefAgent()
    query = "Peux tu me donner une recette pour un plat français avec des ingrédients de saison ?"
    print(f"👤 User: {query}")
    print(f"🤖 Agent: {agent.ask_chef(query)}")
    agent.langfuse.flush()