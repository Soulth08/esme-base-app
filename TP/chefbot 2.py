import os
import json
import re
import time # Import indispensable pour le sleep
from dotenv import load_dotenv
from langfuse import observe, get_client, propagate_attributes
import litellm

# Chargement des variables d'environnement
load_dotenv()

MODEL_ID = "groq/llama-3.1-8b-instant"

def ask_chef(system_prompt: str, user_prompt: str, temperature: float = 0.3):
    # On place le sleep ICI pour qu'il s'applique à chaque appel
    print(f"Attente de 5s (Rate Limit)...")
    time.sleep(5)
    
    response = litellm.completion(
        model=MODEL_ID,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature,
        api_key=os.getenv("GROQ_API_KEY")
    )
    return response.choices[0].message.content

@observe(name="1. Planification (JSON)")
def get_planning_steps(constraints: str):
    langfuse = get_client()
    system_prompt = "Tu es un assistant chef. Décompose la création d'un menu d'une semaine journalier en 3 étapes. Réponds UNIQUEMENT en JSON. Format: ['étape 1', 'étape 2', 'étape 3']"
    
    attempts = 0
    while attempts < 2:
        res = ask_chef(system_prompt, f"Contraintes : {constraints}")
        try:
            json_str = re.search(r'\[.*\]', res, re.DOTALL).group() if "[" in res else res
            return json.loads(json_str)

            
        except Exception as e:
            attempts += 1
            if attempts == 2:
                langfuse.update_current_observation(
                    level="ERROR",
                    status_message=f"Échec parsing JSON: {str(e)}"
                )
                return ["Identifier les ingrédients", "Structurer les repas", "Finaliser le menu"]
    return []

@observe(name="2. Exécution des étapes")
def execute_steps(steps: list, constraints: str):
    results = []
    current_context = f"Contraintes : {constraints}"
    
    for step in steps:
        prompt = f"Exécute cette étape : {step}. Contexte actuel : {current_context}"
        res = ask_chef("Tu es ChefBot.", prompt)
        results.append(res)
        current_context += f"\n- {step}: {res}"
        
    return results

@observe(name="3. Synthèse Finale")
def synthesize_menu(work_done: list):
    # Le sleep de ask_chef s'appliquera aussi ici
    return ask_chef("Tu es ChefBot.", f"Compile ces éléments en un menu pour une semaine entière, jour par jour: {str(work_done)}")

@observe(name="Planification Menu Hebdomadaire")
def plan_weekly_menu(constraints: str) -> str:
    with propagate_attributes(tags=["COLPIN / MORETTI", "Partie 2"]):
        steps = get_planning_steps(constraints)
        details = execute_steps(steps, constraints)
        return synthesize_menu(details)

if __name__ == "__main__":
    langfuse = get_client()
    contraintes = "Végétarien, budget étudiant, produits d'hiver."
    
    print(f"🚀 Lancement du planning (Mode: Anti Rate Limit)")
    try:
        menu = plan_weekly_menu(contraintes)
        print("\n--- MENU FINAL ---\n", menu)
    finally:
        langfuse.flush()