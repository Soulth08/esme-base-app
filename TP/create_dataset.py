import os
from dotenv import load_dotenv
from langfuse import Langfuse

# Charger les variables d'environnement
load_dotenv()

DATASET_NAME = "chefbot-menu-eval-COLPIN-MORETTI"


langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
)

def setup_evaluation_dataset():
    print(f"Création du dataset '{DATASET_NAME}'...")

    langfuse.create_dataset(
        name=DATASET_NAME,
        description="Dataset d'évaluation pour le ChefBot",
        metadata={"authors": "COLPIN / MORETTI"}
    )
    
    test_cases = [
        {
            "input": {"constraints": "Repas pour diabétique"},
            "expected": {"must_avoid": ["sucre", "miel", "sirop", "pâtes blanches"], "must_include": ["légumes verts", "fibres"]}
        },
        {
            "input": {"constraints": "Étudiant sans four avec petit budget"},
            "expected": {"must_avoid": ["rôti", "gratin", "pizza"], "must_include": ["pâtes", "riz", "poêle"]}
        },
        {
            "input": {"constraints": "Sportif en prise de masse, sans produits laitiers"},
            "expected": {"must_avoid": ["beurre", "fromage", "lait"], "must_include": ["poulet", "œufs", "tofu", "riz"]}
        },
        {
            "input": {"constraints": "Menu de Noël traditionnel mais végétarien"},
            "expected": {"must_avoid": ["dinde", "foie gras", "saumon"], "must_include": ["marrons", "champignons", "truffe"]}
        },
        {
            "input": {"constraints": "Régime paléo strict"},
            "expected": {"must_avoid": ["pâtes", "pain", "riz", "légumineuses"], "must_include": ["viande", "noix", "baies"]}
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        langfuse.create_dataset_item(
            dataset_name=DATASET_NAME,
            input=case["input"],
            expected_output=case["expected"]
        )
        print(f"  ✓ Item {i}/{len(test_cases)} ajouté")
    
    print(f"✅ Dataset créé avec {len(test_cases)} items.")
    langfuse.flush()

if __name__ == "__main__":
    setup_evaluation_dataset()