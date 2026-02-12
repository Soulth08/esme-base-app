import os
import json
import sys
from datetime import datetime
from dotenv import load_dotenv
import litellm
from smolagents import CodeAgent, LiteLLMModel, tool, Tool

# --- CONFIGURATION ET UTILITAIRE DE TRACING ---

load_dotenv()
MODEL_ID = "groq/llama-3.3-70b-versatile"

if not os.getenv("GROQ_API_KEY"):
    print("ATTENTION : La variable GROQ_API_KEY est manquante dans le fichier .env")

# Classe utilitaire pour enregistrer la sortie dans un fichier
class DualLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Activation du logging vers fichier
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
sys.stdout = DualLogger(f"run_{timestamp}.txt")



# PARTIE 4 : RECETTES & FRIGO


FRIDGE_CONTENT = ["poulet", "crème fraîche", "champignons", "pâtes", "épinards", "fromage"]

RECIPES_DB = {
    "poulet aux champignons": "1. Saisir le poulet. 2. Ajouter champignons et crème. 3. Servir avec des pâtes.",
    "pâtes aux épinards": "1. Cuire les pâtes. 2. Faire revenir les épinards avec de la crème. 3. Mélanger.",
    "gratin de pâtes": "1. Cuire les pâtes. 2. Mettre dans un plat avec crème et fromage. 3. Gratiner au four."
}

DIETARY_DB = {
    "crème fraîche": "Contient du lactose. Riche en lipides.",
    "poulet": "Riche en protéines. Viande.",
    "champignons": "Légume. Riche en fibres.",
    "pâtes": "Féculent. Contient du gluten.",
    "épinards": "Légume. Riche en fer."
}

# --- Outils Partie 4 (CORRIGÉS) ---

@tool
def check_fridge_tool() -> str:
    """
    Vérifie les ingrédients disponibles dans le frigo.
    Retourne une liste sous forme de chaîne de caractères.
    """
    return str(FRIDGE_CONTENT)

@tool
def get_recipe_tool(dish_name: str) -> str:
    """
    Trouve une recette pour un plat donné.
    
    Args:
        dish_name: Le nom du plat recherché (ex: 'poulet aux champignons').
    """
    for key in RECIPES_DB:
        if dish_name.lower() in key.lower() or key.lower() in dish_name.lower():
            return RECIPES_DB[key]
    return "Recette introuvable."

@tool
def check_dietary_info_tool(ingredient: str) -> str:
    """
    Donne les informations nutritionnelles et allergènes d'un ingrédient.
    
    Args:
        ingredient: Le nom de l'ingrédient à vérifier.
    """
    return DIETARY_DB.get(ingredient.lower(), "Info inconnue.")

def run_partie_4():
    print("\n\nPARTIE 4 : FRIGO & RECETTES ")
    model = LiteLLMModel(model_id=MODEL_ID)
    agent = CodeAgent(
        tools=[check_fridge_tool, get_recipe_tool, check_dietary_info_tool], 
        model=model,
        add_base_tools=False
    )
    query = "Qu'est-ce que j'ai dans le frigo ? Donne moi une recette possible avec ces ingrédients, et vérifie si la recette contient du lactose."
    
    try:
        agent.run(query)
    except Exception as e:
        print(f"Erreur Partie 4: {e}")



# PARTIE 5 : LE RESTAURANT INTELLIGENT


# 5.1 - Outil Base de Données Tool est dans la parentese pas avec le @tool

class MenuDatabaseTool(Tool):
    name = "menu_search"
    description = "Recherche des plats dans le menu selon des critères."
    inputs = {
        "category": {"type": "string", "description": "Optionnel: 'Entrée', 'Plat', 'Dessert'", "nullable": True},
        "max_price": {"type": "integer", "description": "Optionnel: prix maximum en euros", "nullable": True},
        "allergen_free": {"type": "string", "description": "Optionnel: allergène à éviter (ex: 'gluten', 'lactose')", "nullable": True}
    }
    output_type = "string"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Base de données de 10 plats minimum
        self.menu_db = [
            {"nom": "Salade César", "prix": 12, "allergenes": ["lactose", "gluten"], "categorie": "Entrée"},
            {"nom": "Soupe de Potiron", "prix": 8, "allergenes": [], "categorie": "Entrée"},
            {"nom": "Carpaccio de Bœuf", "prix": 14, "allergenes": [], "categorie": "Entrée"},
            {"nom": "Risotto aux Champignons", "prix": 18, "allergenes": ["lactose"], "categorie": "Plat"},
            {"nom": "Steak Frites", "prix": 22, "allergenes": [], "categorie": "Plat"},
            {"nom": "Curry de Légumes", "prix": 16, "allergenes": [], "categorie": "Plat"}, # Vegan
            {"nom": "Pâtes Carbonara", "prix": 17, "allergenes": ["lactose", "gluten", "oeuf"], "categorie": "Plat"},
            {"nom": "Pavé de Saumon", "prix": 20, "allergenes": ["poisson"], "categorie": "Plat"},
            {"nom": "Mousse au Chocolat", "prix": 7, "allergenes": ["lactose", "oeuf"], "categorie": "Dessert"},
            {"nom": "Salade de Fruits", "prix": 6, "allergenes": [], "categorie": "Dessert"}
        ]

    def forward(self, category: str = None, max_price: int = None, allergen_free: str = None) -> str:
        results = self.menu_db
        
        # Filtrage
        if category:
            results = [p for p in results if p["categorie"].lower() == category.lower()]
        
        if max_price:
            results = [p for p in results if p["prix"] <= max_price]
            
        if allergen_free:
            # On garde le plat si l'allergène N'EST PAS dans la liste des allergènes du plat
            results = [p for p in results if allergen_free.lower() not in [a.lower() for a in p["allergenes"]]]
            
        if not results:
            return "Aucun plat trouvé avec ces critères."
            
        return json.dumps(results, ensure_ascii=False)

# Outil de calcul pour l'agent (5.2)
#utilisation de l'outils de calcu
@tool
def calculate_bill(prices: list[int]) -> int:
    """
    Calcule la somme totale d'une liste de prix.
    
    Args:
        prices: Liste des prix des plats (entiers).
    """
    return sum(prices)

# 5.2 - Agent avec Planification

def run_partie_5_planning():
    print("\n\nPARTIE 5.2 : AGENT PLANIFICATEUR ")
    
    model = LiteLLMModel(model_id=MODEL_ID)
    menu_tool = MenuDatabaseTool()
    
    # Initialisation de l'agent avec planning_interval
    agent = CodeAgent(
        tools=[menu_tool, calculate_bill],
        model=model,
        planning_interval=2, # Planifie toutes les 2 étapes (donnée dans l'énoncé)
        add_base_tools=True
    )

    query = """
    On est 3 clients au restaurant.
    1. Un végétarien (pas de viande/poisson).
    2. Un sans gluten (pas de gluten).
    3. Moi je mange de tout.
    
    Budget TOTAL max pour le groupe : 60 euros.
    
    Propose-nous un menu complet (1 plat par personne) qui respecte le budget et les régimes.
    Utilise l'outil de calcul pour vérifier le total.
    """
    
    try:
        agent.run(query)
    except Exception as e:
        print(f"Erreur Planning: {e}")

# 5.3 - Agent Conversationnel Multi-tours

def run_partie_5_conversation():
    print("\n\nPARTIE 5.3 : AGENT CONVERSATIONNEL ")
    
    model = LiteLLMModel(model_id=MODEL_ID)
    menu_tool = MenuDatabaseTool()
    
    agent = CodeAgent(
        tools=[menu_tool, calculate_bill],
        model=model,
        add_base_tools=True
    )

    # Simulation du dialogue
    dialogue_turns = [
        "Bonjour, quelles sont vos entrées sans allergènes (sans rien) ?",
        "Je vais plutôt prendre un plat. Avez-vous quelque chose de végétarien ?",
        "Ok pour le Curry de Légumes. Apportez-moi ça et l'addition svp."
    ]

    for i, turn in enumerate(dialogue_turns):
        print(f"\n--- TOUR {i+1} ---")
        print(f"Client: {turn}")
        # reset=False est crucial ici pour garder la mémoire de la conversation
        agent.run(turn, reset=False)



# MAIN EXECUTION


if __name__ == "__main__":
    #regarde la console si pb sur litellm
    litellm._turn_on_debug()
    
    
    run_partie_5_planning()
    run_partie_5_conversation()
    
    print(f"\n\n--- Fin du programme. Trace sauvegardée dans le fichier txt généré. ---")