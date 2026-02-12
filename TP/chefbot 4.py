#On a pas utiliser le dataset car on arrive a le cree ais il y a une une limite donc on a pas fait 3.2et 3.3 , le dataset ici est une base de donnee simulee
import os
import json
import time
from dotenv import load_dotenv
import litellm
from smolagents import CodeAgent, LiteLLMModel, tool

# 1. Chargement des variables d'environnement
load_dotenv()


MODEL_ID = "groq/llama-3.3-70b-versatile" 

# Vérification basique de la clé API
if not os.getenv("GROQ_API_KEY"):
    print("ATTENTION : La variable GROQ_API_KEY est manquante dans le fichier .env")


# DONNÉES SIMULÉES (Base de données)


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


# PARTIE 4.1 : FONCTIONS PYTHON BRUTES


def check_fridge():
    """Retourne la liste des ingrédients disponibles."""
    return str(FRIDGE_CONTENT)

def get_recipe(dish_name: str):
    """Retourne une recette détaillée pour un plat donné."""
    # Petite sécurité pour gérer les majuscules/minuscules
    for key in RECIPES_DB:
        if dish_name.lower() in key.lower() or key.lower() in dish_name.lower():
            return RECIPES_DB[key]
    return "Recette non trouvée."

def check_dietary_info(ingredient: str):
    """Retourne les infos nutritionnelles et allergènes."""
    return DIETARY_DB.get(ingredient.lower(), "Info non disponible.")


# PARTIE 4.2 : BOUCLE DE TOOL CALLING MANUELLE


def run_manual_loop(user_query):
    print(f"\n--- 🔧 DÉMARRAGE MODE MANUEL ({MODEL_ID}) ---")
    
    # Définition du schéma pour le LLM
    tools_schema = [
        {
            "type": "function",
            "function": {
                "name": "check_fridge",
                "description": "Vérifie les ingrédients disponibles dans le frigo.",
                "parameters": {"type": "object", "properties": {}}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_recipe",
                "description": "Donne la recette d'un plat spécifique.",
                "parameters": {
                    "type": "object",
                    "properties": {"dish_name": {"type": "string", "description": "Le nom du plat"}},
                    "required": ["dish_name"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "check_dietary_info",
                "description": "Donne les informations nutritionnelles d'un ingrédient.",
                "parameters": {
                    "type": "object",
                    "properties": {"ingredient": {"type": "string"}},
                    "required": ["ingredient"]
                }
            }
        }
    ]

    # Ajout d'un System Prompt pour stabiliser le modèle
    messages = [
        {"role": "system", "content": "Tu es un assistant culinaire utile. Si tu as besoin d'info, utilise les outils fournis. Réponds en Français."},
        {"role": "user", "content": user_query}
    ]
    
    # Boucle d'exécution (Max 5 itérations)
    for i in range(5):
        print(f"Itération {i+1}...")
        
        try:
            response = litellm.completion(
                model=MODEL_ID,
                messages=messages,
                tools=tools_schema,
                tool_choice="auto"
            )
        except Exception as e:
            print(f"Erreur critique API : {e}")
            return

        message = response.choices[0].message
        tool_calls = message.tool_calls

        # Cas 1 : Le LLM répond directement sans outil
        if not tool_calls:
            print(f"Réponse finale : {message.content}")
            return message.content

        # Cas 2 : Le LLM veut utiliser des outils
        messages.append(message) # On ajoute la demande du LLM à l'historique

        for tool_call in tool_calls:
            function_name = tool_call.function.name
            # Gestion d'erreur si les arguments sont mal formés
            try:
                args = json.loads(tool_call.function.arguments)
            except:
                args = {}
            
            call_id = tool_call.id
            
            print(f"Appel détecté : {function_name} | Args: {args}")

            # Exécution manuelle
            result = "Erreur"
            if function_name == "check_fridge":
                result = check_fridge()
            elif function_name == "get_recipe":
                result = get_recipe(args.get("dish_name", ""))
            elif function_name == "check_dietary_info":
                result = check_dietary_info(args.get("ingredient", ""))
            
            # Injection du résultat dans l'historique
            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "name": function_name,
                "content": str(result)
            })


# PARTIE 4.3 : SMOLAGENTS (FRAMEWORK)


# 1. Définition des outils avec @tool
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
    # Réutilisation de la logique précédente
    for key in RECIPES_DB:
        if dish_name.lower() in key.lower() or key.lower() in dish_name.lower():
            return RECIPES_DB[key]
    return "Recette introuvable."

@tool
def check_dietary_info_tool(ingredient: str) -> str:
    """
    Donne les informations nutritionnelles et allergènes d'un ingrédient.
    Args:
        ingredient: Le nom de l'ingrédient.
    """
    return DIETARY_DB.get(ingredient.lower(), "Info inconnue.")

def run_smolagents_loop(user_query):
    print(f"\n\n--- DÉMARRAGE MODE SMOLAGENTS ({MODEL_ID}) ---")
    
    # Initialisation du modèle
    model = LiteLLMModel(model_id=MODEL_ID)

    # Création de l'agent
    agent = CodeAgent(
        tools=[check_fridge_tool, get_recipe_tool, check_dietary_info_tool], 
        model=model,
        add_base_tools=False
    )

    # Exécution
    try:
        agent_result = agent.run(user_query)
        print(f"\nRésultat Smolagents : {agent_result}")
    except Exception as e:
        print(f"Erreur Smolagents : {e}")


# MAIN EXECUTION


if __name__ == "__main__":
    # La question complexe qui nécessite plusieurs outils
    QUERY = "Qu'est-ce que j'ai dans le frigo ? Donne moi une recette possible avec ces ingrédients, et vérifie si la recette contient du lactose."
    
    # 1. Lancer la méthode manuelle
    run_manual_loop(QUERY)
    
    # 2. Lancer la méthode Framework
    run_smolagents_loop(QUERY)