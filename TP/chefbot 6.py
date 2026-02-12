#basé sur votre TP 10_agent_conversationnel.py très utile
import os
import json
import sys
from datetime import datetime
from dotenv import load_dotenv
from smolagents import CodeAgent, tool, Tool
from smolagents.models import LiteLLMModel


# CONFIGURATION


load_dotenv()


MODEL_ID = "groq/llama-3.3-70b-versatile"

if not os.getenv("GROQ_API_KEY"):
    print("ERREUR: GROQ_API_KEY manquante dans .env")
    sys.exit(1)


# DONNÉES


FRIDGE_CONTENT = ["poulet", "crème fraîche", "champignons", "pâtes", "épinards", "fromage"]

RECIPES_DB = {
    "poulet aux champignons": "1. Saisir le poulet. 2. Ajouter champignons et crème. 3. Servir avec pâtes.",
    "pâtes aux épinards": "1. Cuire les pâtes. 2. Faire revenir épinards avec crème. 3. Mélanger.",
    "gratin de pâtes": "1. Cuire pâtes. 2. Mettre dans plat avec crème et fromage. 3. Gratiner."
}

DIETARY_DB = {
    "crème fraîche": "Contient lactose. Riche en lipides.",
    "poulet": "Riche en protéines. Viande.",
    "champignons": "Légume. Riche en fibres.",
    "pâtes": "Féculent. Contient gluten.",
    "épinards": "Légume. Riche en fer.",
    "fromage": "Contient lactose.",
    "olive": "Fruit à coque.",
    "quinoa": "Sans gluten.",
    "potiron": "Légume.",
    "boeuf": "Viande.",
    "saumon": "Poisson.",
    "chèvre": "Contient lactose."
}


# OUTILS

#ingédients disponible
@tool
def check_fridge_tool() -> str:
    """Vérifie les ingrédients disponibles dans le frigo."""
    return f"Ingrédients: {', '.join(FRIDGE_CONTENT)}"

@tool
def get_recipe_tool(dish_name: str) -> str:
    """Trouve une recette pour un plat.
    
    Args:
        dish_name: Nom du plat recherché
    """
    for key in RECIPES_DB:
        if dish_name.lower() in key.lower():
            return f"Recette '{key}': {RECIPES_DB[key]}"
    return f"Recette introuvable. Disponibles: {', '.join(RECIPES_DB.keys())}"

@tool
def check_dietary_info_tool(ingredient: str) -> str:
    """Infos nutritionnelles et allergènes d'un ingrédient.
    
    Args:
        ingredient: Nom de l'ingrédient
    """
    return DIETARY_DB.get(ingredient.lower(), f"Info inconnue pour '{ingredient}'")

class MenuDatabaseTool(Tool):
    """Recherche dans le menu du restaurant."""
    name = "menu_search"
    description = "Recherche plats selon catégorie, prix, allergènes"
    inputs = {
        "category": {"type": "string", "description": "Apéritif/Entrée/Plat/Dessert", "nullable": True},
        "max_price": {"type": "integer", "description": "Prix max en euros", "nullable": True},
        "allergen_free": {"type": "string", "description": "Allergène à éviter", "nullable": True}
    }
    output_type = "string"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.menu_db = [
            # Apéritifs
            {"nom": "Tapenade d'Olives", "prix": 4, "allergenes": ["fruits à coque"], "categorie": "Apéritif"},
            {"nom": "Mini-Toasts Chèvre", "prix": 5, "allergenes": ["lactose", "gluten"], "categorie": "Apéritif"},
            {"nom": "Bâtonnets de Légumes", "prix": 3, "allergenes": [], "categorie": "Apéritif"},
            
            # Entrées
            {"nom": "Salade César", "prix": 12, "allergenes": ["lactose", "gluten"], "categorie": "Entrée"},
            {"nom": "Soupe de Potiron", "prix": 8, "allergenes": [], "categorie": "Entrée"},
            {"nom": "Carpaccio de Bœuf", "prix": 14, "allergenes": ["viande"], "categorie": "Entrée"},
            {"nom": "Salade Quinoa Avocat", "prix": 11, "allergenes": [], "categorie": "Entrée"},
            
            # Plats
            {"nom": "Risotto aux Champignons", "prix": 18, "allergenes": ["lactose"], "categorie": "Plat"},
            {"nom": "Steak Frites", "prix": 22, "allergenes": ["viande"], "categorie": "Plat"},
            {"nom": "Curry de Légumes", "prix": 16, "allergenes": [], "categorie": "Plat"},
            {"nom": "Pâtes Carbonara", "prix": 17, "allergenes": ["lactose", "gluten", "oeuf", "viande"], "categorie": "Plat"},
            {"nom": "Pavé de Saumon", "prix": 20, "allergenes": ["poisson"], "categorie": "Plat"},
            
            # Desserts
            {"nom": "Mousse au Chocolat", "prix": 7, "allergenes": ["lactose", "oeuf"], "categorie": "Dessert"},
            {"nom": "Salade de Fruits", "prix": 6, "allergenes": [], "categorie": "Dessert"},
            {"nom": "Sorbet Citron", "prix": 5, "allergenes": [], "categorie": "Dessert"}
        ]

    def forward(self, category: str = None, max_price: int = None, allergen_free: str = None) -> str:
        results = self.menu_db.copy()
        
        if category:
            results = [p for p in results if p["categorie"].lower() == category.lower()]
        if max_price:
            results = [p for p in results if p["prix"] <= max_price]
        if allergen_free:
            results = [p for p in results if not any(allergen_free.lower() in a.lower() for a in p["allergenes"])]
            
        return json.dumps(results, ensure_ascii=False, indent=2) if results else "Aucun plat trouvé."

@tool
def calculate_bill(prices: list) -> int:
    """Calcule la somme de prix.
    
    Args:
        prices: Liste de prix
    """
    return sum(int(p) for p in prices)


# SYSTÈME MULTI-AGENT


def run_multi_agent_system():
    print("\n" + "="*60)
    print("SYSTÈME MULTI-AGENTS - RESTAURANT")
    print("="*60 + "\n")
    
    # SOLUTION: Créer le modèle avec les bons paramètres
    import litellm
    
    # Configuration explicite pour Groq
    model = LiteLLMModel(
        model_id=MODEL_ID,
        api_key=os.getenv("GROQ_API_KEY")
    )
    
    print(f"OK Modèle: {MODEL_ID}\n")

    # AGENTS SPÉCIALISÉS
    print("--- Création des agents ---\n")
    #agent nutrisioniste
    nutritionist = CodeAgent(
        tools=[check_dietary_info_tool],
        model=model,
        name="nutritionist",
        description="Expert nutrition - vérifie allergènes",
        add_base_tools=False
    )
    print("OK Nutritionist")
#agent chef
    chef = CodeAgent(
        tools=[check_fridge_tool, get_recipe_tool],
        model=model,
        name="chef",
        description="Chef cuisinier - recettes et frigo",
        add_base_tools=False
    )
    print("OK Chef")
#agent calcul cout
    budget_manager = CodeAgent(
        tools=[MenuDatabaseTool(), calculate_bill],
        model=model,
        name="budget_manager",
        description="Gère menu restaurant et budget",
        add_base_tools=False
    )
    print("OK Budget Manager\n")


    # MANAGER (il utilise tous les agents spécialisés)
    print("--- Création du manager ---\n")
    
    manager = CodeAgent(
        tools=[],
        managed_agents=[nutritionist, chef, budget_manager],
        model=model,
        name="manager",
        description="Manager - délègue aux agents spécialisés",
        add_base_tools=False
    )
    print("Manager créé\n")

    # REQUÊTE COMPLEXE du TP    
    query = """
    MISSION:
    8 personnes samedi soir. Budget total: 120 euros.
    
    CONTRAINTES:
    - 2 végétariens (pas viande/poisson)
    - 1 intolérant au gluten
    - 1 allergique aux fruits à coque
    - 4 sans restriction
    
    TÂCHE:
    Menu UNIQUE pour tous (même plats) avec:
    1. Apéritif
    2. Entrée
    3. Plat
    4. Dessert
    
    Compatible avec TOUTES les restrictions (végétarien + sans gluten + sans fruits à coque).
    
    INSTRUCTIONS:
    1. Utilise 'budget_manager' pour chercher plats compatibles
    2. Si doute sur ingrédient, demande au 'nutritionist'
    3. Affiche menu final
    4. Calcule total pour 8 personnes
    5. Vérifie que total <= 120€
    """

    print("="*60)
    print("REQUÊTE CLIENT")
    print("="*60)
    print(query)
    print("="*60 + "\n")
    
    try:
        print(">>> Traitement par le manager...\n")
        response = manager.run(query)
        
        print("\n" + "="*60)
        print("RÉSULTAT FINAL")
        print("="*60)
        print(response)
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\nErreur: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- Terminé ---")



if __name__ == "__main__":
    #on croise les doights pour que ca marche
    run_multi_agent_system()