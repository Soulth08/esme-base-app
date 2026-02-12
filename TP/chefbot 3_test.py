import os
import json
import re
import time
from dotenv import load_dotenv
from langfuse import observe, get_client, propagate_attributes
import litellm

# 1. Configuration initiale
load_dotenv()
MODEL_ID = "groq/llama-3.3-70b-versatile"
DATASET_NAME = "chefbot-menu-eval-COLPIN-MORETTI"
langfuse = get_client()

# --- FONCTION DE BASE AVEC ANTI-RATE LIMIT ---

def ask_chef(system_prompt: str, user_prompt: str, temperature: float = 0.3):
    # Sécurité indispensable pour Groq 8b-instant (surtout pendant une expérience)
    print(f"Attente 5s (Rate Limit)...")
    time.sleep(20)
    
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

# --- PARTIE 2 : LOGIQUE DE PLANNING (LES ÉTAPES) ---

@observe(name="1. Planification (JSON)")
def get_planning_steps(constraints: str):
    system_prompt = "Tu es un assistant chef. Décompose la création d'un menu hebdomadaire en 3 étapes. Réponds UNIQUEMENT en JSON. Format: ['étape 1', 'étape 2', 'étape 3']"
    
    attempts = 0
    while attempts < 2:
        res = ask_chef(system_prompt, f"Contraintes : {constraints}")
        try:
            json_str = re.search(r'\[.*\]', res, re.DOTALL).group() if "[" in res else res
            return json.loads(json_str)
        except Exception as e:
            attempts += 1
            if attempts == 2:
                langfuse.update_current_observation(level="ERROR", status_message=f"Erreur JSON: {str(e)}")
                return ["Identifier ingrédients", "Structurer repas", "Finaliser"]
    return []

@observe(name="2. Exécution des étapes")
def execute_steps(steps: list, constraints: str):
    results = []
    current_context = f"Contraintes : {constraints}"
    for step in steps:
        res = ask_chef("Tu es ChefBot.", f"Fais l'étape: {step}. Contexte: {current_context}")
        results.append(res)
        current_context += f"\n- {step}: {res}"
    return results

@observe(name="3. Synthèse Finale")
def synthesize_menu(work_done: list):
    return ask_chef("Tu es ChefBot.", f"Compile ces travaux en menu hebdomadaire complet : {str(work_done)}")

@observe(name="Planification Menu Hebdomadaire")
def plan_weekly_menu(constraints: str) -> str:
    # On garde tes tags pour identifier le groupe
    with propagate_attributes(tags=["COLPIN / MORETTI", "Partie 3"]):
        steps = get_planning_steps(constraints)
        details = execute_steps(steps, constraints)
        return synthesize_menu(details)

# --- PARTIE 3 : ÉVALUATEURS ET EXPÉRIENCE ---

def rule_evaluator(output: str, expected: dict, **kwargs):
    """Vérification stricte des ingrédients requis/interdits"""
    # On récupère 'input' si besoin via kwargs.get('input')
    output_lower = output.lower()
    
    # Check Interdits (must_avoid)
    forbidden = [w for w in expected.get("must_avoid", []) if w.lower() in output_lower]
    avoid_score = 1 if not forbidden else 0
    
    # Check Requis (must_include)
    required = expected.get("must_include", [])
    found = sum(1 for w in required if w.lower() in output_lower)
    include_score = found / len(required) if required else 1
    
    return [
        {"name": "rule_avoid_forbidden", "value": avoid_score},
        {"name": "rule_include_required", "value": include_score}
    ]

def llm_judge(output: str, expected: dict, **kwargs):
    """Juge LLM pour la qualité subjective"""
    # Langfuse nous envoie l'input dans kwargs
    input_data = kwargs.get('input', {})
    constraints = input_data.get('constraints', "Non spécifiées")

    prompt = f"""Juge ce menu sur 3 critères (0.0 à 1.0).
    Contraintes initiales de l'utilisateur: {constraints}
    Menu généré par le chef: {output}
    
    Réponds UNIQUEMENT en JSON: {{"pertinence": x, "creativite": x, "praticite": x}}"""
    
    res = ask_chef("Tu es un critique culinaire expert.", prompt, temperature=0)
    try:
        # Extraction du JSON dans la réponse
        match = re.search(r'\{.*\}', res, re.DOTALL)
        scores = json.loads(match.group()) if match else json.loads(res)
        return [{"name": f"llm_{k}", "value": v} for k, v in scores.items()]
    except Exception as e:
        print(f"⚠️ Erreur juge LLM: {e}")
        return [{"name": "llm_eval_error", "value": 0}]

def run_chef_experiment():
    print(f"Partie 3 - Lancement de l'expérience sur : {DATASET_NAME}")
    
    # 1. On récupère les items du dataset manuellement pour éviter l'erreur de len()
    dataset = langfuse.get_dataset(DATASET_NAME)
    
    # 2. On définit la fonction de test
    # Note : Langfuse passe l'objet 'item' à la fonction task
    def my_chef_task(item):
        # On extrait la contrainte depuis l'input de l'item
        constraints = item.input["constraints"]
        return plan_weekly_menu(constraints)

    # 3. Lancement de l'expérience
    # On utilise 'data' pour les items et 'task' pour la fonction
    result = langfuse.run_experiment(
        name="ChefBot_Full_Workflow_V3",
        data=dataset.items,       # ✅ On passe la liste des items (.items)
        task=my_chef_task,        # ✅ On utilise 'task' au lieu de 'run'
        evaluators=[rule_evaluator, llm_judge]
    )
    
    print("\n" + "="*50)
    print("✅ Expérience terminée ! Regarde tes scores sur Langfuse.")
    print("="*50)

if __name__ == "__main__":
    run_chef_experiment()
    langfuse.flush()