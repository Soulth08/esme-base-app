
import os
import json
from datetime import datetime
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
from dotenv import load_dotenv

from langfuse import Langfuse

import litellm
from smolagents import CodeAgent, tool, Tool
from smolagents.models import LiteLLMModel

load_dotenv()


# CONFIGURATION


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

MODEL_CONFIG_1 = "groq/llama-3.3-70b-versatile"  
MODEL_CONFIG_2 = "groq/llama-3.3-70b-versatile"      


# 7.1 - DATASET DE SCÉNARIOS


@dataclass
class ExpectedOutput:
    must_respect: List[str]
    expected_services: int
    max_budget: float

@dataclass
class Scenario:
    id: str
    difficulty: str
    description: str
    query: str
    expected_output: ExpectedOutput
    
    def to_dict(self):
        return {
            "id": self.id,
            "difficulty": self.difficulty,
            "description": self.description,
            "query": self.query,
            "expected_output": asdict(self.expected_output)
        }

EVALUATION_DATASET = [
    #création de tout ls sénarios
    # Scénario 1: FACILE
    Scenario(
        id="scenario_1_facile",
        difficulty="Facile",
        description="Dîner simple 2 personnes",
        query="""
        Dîner romantique pour 2 personnes.
        Pas de restrictions alimentaires.
        Budget: 60€.
        Je veux entrée, plat, dessert.
        """,
        expected_output=ExpectedOutput(
            must_respect=["2 personnes", "entrée", "plat", "dessert"],
            expected_services=3,
            max_budget=60.0
        )
    ),
    
    # Scénario 2: MOYEN
    Scenario(
        id="scenario_2_moyen",
        difficulty="Moyen",
        description="Repas famille avec allergie",
        query="""
        Repas famille 4 personnes.
        1 personne allergique lactose.
        Budget: 100€.
        Menu unique: apéritif, entrée, plat, dessert.
        """,
        expected_output=ExpectedOutput(
            must_respect=["4 personnes", "sans lactose", "apéritif", "entrée", "plat", "dessert"],
            expected_services=4,
            max_budget=100.0
        )
    ),
    
    # Scénario 3: DIFFICILE
    Scenario(
        id="scenario_3_difficile",
        difficulty="Difficile",
        description="Dîner multi-contraintes budget serré",
        query="""
        Dîner 6 personnes.
        - 2 végétariens
        - 1 intolérant gluten
        - 1 allergique fruits à coque
        Budget SERRÉ: 90€.
        Menu unique: apéritif, entrée, plat, dessert.
        Compatible avec TOUTES les restrictions.
        """,
        expected_output=ExpectedOutput(
            must_respect=["6 personnes", "végétarien", "sans gluten", "sans fruits à coque",
                         "apéritif", "entrée", "plat", "dessert"],
            expected_services=4,
            max_budget=90.0
        )
    ),
    
    # Scénario 4: EXTRÊME
    Scenario(
        id="scenario_4_extreme",
        difficulty="Extreme",
        description="Événement contraintes multiples",
        query="""
        Événement 12 personnes.
        - 3 végétariens stricts
        - 2 personnes hallal
        - 1 diabétique
        - 1 intolérant gluten
        - 2 allergiques lactose
        - 1 allergique fruits de mer
        Budget: 150€ TOTAL.
        Menu unique: apéritif, entrée, plat, dessert.
        Respecter TOUTES les contraintes simultanément.
        """,
        expected_output=ExpectedOutput(
            must_respect=["12 personnes", "végétarien", "sans gluten", "sans lactose",
                         "sans fruits de mer", "diabète-friendly",
                         "apéritif", "entrée", "plat", "dessert"],
            expected_services=4,
            max_budget=150.0
        )
    )
]


# OUTILS DU SYSTÈME MULTI-AGENT


DIETARY_DB = {
    "olive": "Fruit à coque", "quinoa": "Sans gluten", "potiron": "Faible index glycémique",
    "citron": "Faible en sucre", "champignons": "Légume", "épinards": "Légume"
}

@tool
def check_dietary_info_tool(ingredient: str) -> str:
    """Infos nutritionnelles."""
    return DIETARY_DB.get(ingredient.lower(), f"Info inconnue")

class MenuDatabaseTool(Tool):
    name = "menu_search"
    description = "Recherche plats restaurant"
    inputs = {
        "category": {"type": "string", "nullable": True},
        "max_price": {"type": "integer", "nullable": True},
        "allergen_free": {"type": "string", "nullable": True}
    }
    output_type = "string"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.menu_db = [
            {"nom": "Bâtonnets Légumes", "prix": 3, "allergenes": [], "categorie": "Apéritif"},
            {"nom": "Soupe Potiron", "prix": 8, "allergenes": [], "categorie": "Entrée"},
            {"nom": "Salade Quinoa", "prix": 11, "allergenes": [], "categorie": "Entrée"},
            {"nom": "Curry Légumes", "prix": 16, "allergenes": [], "categorie": "Plat"},
            {"nom": "Sorbet Citron", "prix": 5, "allergenes": [], "categorie": "Dessert"},
            {"nom": "Salade Fruits", "prix": 6, "allergenes": [], "categorie": "Dessert"},
        ]

    def forward(self, category=None, max_price=None, allergen_free=None):
        results = self.menu_db.copy()
        if category:
            results = [p for p in results if p["categorie"].lower() == category.lower()]
        if max_price:
            results = [p for p in results if p["prix"] <= max_price]
        if allergen_free:
            results = [p for p in results if allergen_free.lower() not in [a.lower() for a in p["allergenes"]]]
        return json.dumps(results, ensure_ascii=False) if results else "Aucun plat"

@tool
#tool permettant le calcul
def calculate_bill(prices: list) -> int:
    """Calcule total."""
    return sum(int(p) for p in prices)


# SYSTÈME MULTI-AGENT

#créations des agents qui utilse tout les tools
def create_multi_agent_system(model_id: str, config_name: str):
    model = LiteLLMModel(model_id=model_id, api_key=GROQ_API_KEY)
    
    nutritionist = CodeAgent(
        tools=[check_dietary_info_tool], model=model,
        name="nutritionist", description="Expert nutrition", add_base_tools=False
    )
    
    budget_manager = CodeAgent(
        tools=[MenuDatabaseTool(), calculate_bill], model=model,
        name="budget_manager", description="Gère menu et budget", add_base_tools=False
    )
    
    manager = CodeAgent(
        tools=[], managed_agents=[nutritionist, budget_manager], model=model,
        name="manager", description="Manager - délègue aux agents", add_base_tools=False
    )
    
    return manager


# 7.2 - JUGE LLM SUR 5 CRITÈRES


@dataclass
class EvaluationCriteria:
    score: float
    reasoning: str
    details: str

#critere de l'évaluation donné dans le TP
@dataclass
class EvaluationResult:
    scenario_id: str
    respect_contraintes: EvaluationCriteria
    completude: EvaluationCriteria
    budget: EvaluationCriteria
    coherence: EvaluationCriteria
    faisabilite: EvaluationCriteria
    average_score: float
    
    #dictionnaire de l'objet
    def to_dict(self):
        return {
            "scenario_id": self.scenario_id,
            "respect_contraintes": asdict(self.respect_contraintes),
            "completude": asdict(self.completude),
            "budget": asdict(self.budget),
            "coherence": asdict(self.coherence),
            "faisabilite": asdict(self.faisabilite),
            "average_score": self.average_score
        }

#utilisation d'un autre llm pour juger verasité (on prend un autre modele)
class LLMJudge:
    def __init__(self, model_id="groq/llama-3.1-70b-versatile"):
        self.model_id = model_id
    
    #evaluation de l'agent via score
    def evaluate(self, scenario: Scenario, agent_response: str) -> EvaluationResult:
        # Évaluer les 5 critères
        respect = self._eval_criterion("respect_contraintes",
            f"Contraintes: {scenario.expected_output.must_respect}\nRéponse: {agent_response}\n"
            f"Toutes respectées? Score 0-1")
        
        completude = self._eval_criterion("completude",
            f"Services attendus: {scenario.expected_output.expected_services}\nRéponse: {agent_response}\n"
            f"Tous présents? Score 0-1")
        
        budget = self._eval_criterion("budget",
            f"Budget max: {scenario.expected_output.max_budget}€\nRéponse: {agent_response}\n"
            f"Respecté? Score 0-1")
        
        coherence = self._eval_criterion("coherence",
            f"Réponse: {agent_response}\nMenu cohérent? Score 0-1")
        
        faisabilite = self._eval_criterion("faisabilite",
            f"Réponse: {agent_response}\nRéalisable amateur? Score 0-1")
        
        avg = (respect.score + completude.score + budget.score + coherence.score + faisabilite.score) / 5
        
        return EvaluationResult(scenario.id, respect, completude, budget, coherence, faisabilite, avg)
    
    def _eval_criterion(self, name: str, question: str) -> EvaluationCriteria:
        prompt = f"{question}\n\nRéponds en JSON: {{\"score\": 0-1, \"reasoning\": \"...\", \"details\": \"...\"}}"
        
        try:
            response = litellm.completion(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2, max_tokens=500
            )
            content = response.choices[0].message.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            result = json.loads(content.strip())
            return EvaluationCriteria(
                score=float(result.get("score", 0)),
                reasoning=result.get("reasoning", ""),
                details=result.get("details", "")
            )
        except Exception as e:
            return EvaluationCriteria(0.0, f"Erreur: {e}", "")


# 7.3 - EXPÉRIMENTATION ET COMPARAISON


@observe()
def run_experiment(scenario: Scenario, model_id: str, config_name: str, langfuse_client: Langfuse):
    print(f"\n{'='*60}\n{scenario.id} ({scenario.difficulty}) - {model_id}\n{'='*60}")
    
    agent = create_multi_agent_system(model_id, config_name)
    
    start = datetime.now()
    try:
        response = agent.run(scenario.query)
        exec_time = (datetime.now() - start).total_seconds()
        success = True
    except Exception as e:
        response = f"Erreur: {e}"
        exec_time = (datetime.now() - start).total_seconds()
        success = False
    
    print(f"{exec_time:.2f}s")
    
    judge = LLMJudge()
    evaluation = judge.evaluate(scenario, response)
    
    print(f"   Score: {evaluation.average_score:.2f}")
    print(f"   Contraintes: {evaluation.respect_contraintes.score:.2f}")
    print(f"   Complétude: {evaluation.completude.score:.2f}")
    print(f"   Budget: {evaluation.budget.score:.2f}")
    print(f"   Cohérence: {evaluation.coherence.score:.2f}")
    print(f"   Faisabilité: {evaluation.faisabilite.score:.2f}")
    
    return {
        "scenario_id": scenario.id,
        "difficulty": scenario.difficulty,
        "model_id": model_id,
        "config_name": config_name,
        "execution_time": exec_time,
        "success": success,
        "response": response,
        "evaluation": evaluation.to_dict()
    }

def compare_configurations():
    print("\n" + "="*60)
    print("ÉVALUATION END-TO-END MULTI-AGENT")
    print("="*60)
    
    langfuse = Langfuse(
        public_key=LANGFUSE_PUBLIC_KEY,
        secret_key=LANGFUSE_SECRET_KEY,
        host=LANGFUSE_HOST
    )
    
    configs = [
        {"name": "config_1_llama70b", "model_id": MODEL_CONFIG_1, "config_name": "default",
         "description": "Llama 70B - Précision"},
        {"name": "config_2_llama8b", "model_id": MODEL_CONFIG_2, "config_name": "default",
         "description": "Llama 8B - Vitesse"}
    ]
    
    all_results = []
    
    for config in configs:
        print(f"\n{'#'*60}\n{config['name']}: {config['description']}\n{'#'*60}")
        
        for scenario in EVALUATION_DATASET:
            result = run_experiment(scenario, config["model_id"], config["config_name"], langfuse)
            result["config"] = config["name"]
            all_results.append(result)
            
            langfuse.trace(
                name=f"{scenario.id}_{config['name']}",
                metadata={"scenario": scenario.to_dict(), "config": config, "result": result},
                tags=[scenario.difficulty, config["name"]]
            )
    
    # Dataset Langfuse (cela ne marcheara pas)
    for scenario in EVALUATION_DATASET:
        langfuse.create_dataset_item(
            dataset_name="chefbot-multiagent-eval",
            input=scenario.query,
            expected_output=scenario.expected_output.__dict__,
            metadata={"id": scenario.id, "difficulty": scenario.difficulty}
        )
    
    # Analyse
    print("\n" + "="*60 + "\nANALYSE COMPARATIVE\n" + "="*60)
    for config in configs:
        config_results = [r for r in all_results if r["config"] == config["name"]]
        avg_score = sum(r["evaluation"]["average_score"] for r in config_results) / len(config_results)
        avg_time = sum(r["execution_time"] for r in config_results) / len(config_results)
        print(f"\n{config['name']}:")
        print(f"  Score moyen: {avg_score:.2f}")
        print(f"  Temps moyen: {avg_time:.2f}s")
    
    # Sauvegarder
    output = f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output, "w") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nRésultats: {output}")
    
    langfuse.flush()
    return all_results

if __name__ == "__main__":
    print("\nÉvaluation ChefBot Multi-Agent - Partie 7")
    results = compare_configurations()
    print("\nC'est finiiiiiiiiiiiiiiiiii")