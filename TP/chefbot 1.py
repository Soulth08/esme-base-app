import os
from dotenv import load_dotenv
from langfuse import observe, get_client, propagate_attributes
import litellm


# Chargement des variables d'environnement
load_dotenv()

@observe(name="ask_chef_call")
def ask_chef(question: str, temperature: float = 0.7) -> str:
    """
    Appel LLM basique vers Groq avec un profil de Chef spécialisé.
    """
    with propagate_attributes(tags=["COLPIN / MORETTI", "1.3"]):

        # 2. Définition du System Prompt (Consigne 1.1)
        system_prompt = (
            "Tu es ChefBot, un chef cuisinier français renommé, "
            "spécialisé dans la cuisine de saison et les produits locaux."
        )

        # 3. Appel au modèle via LiteLLM (interface pour Groq)
        response = litellm.completion(
            model="groq/llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            temperature=temperature,
            api_key=os.getenv("GROQ_API_KEY")
        )

        return response.choices[0].message.content

if __name__ == "__main__":
    
    langfuse = get_client()
    user_query = "Peux-tu me suggérer un plat français avec des légumes de saison ?"
    
    # 1.3 - Test avec 3 températures différentes
    temperatures = [0.1, 0.7, 1.2, 2.0]
    
    print(f"👤 Question: {user_query}\n")

    for temp in temperatures:
        print(f"--- Test avec Temperature = {temp} ---")
        res = ask_chef(user_query, temperature=temp)
        print(f"🤖 ChefBot: {res}\n")

    # Envoi final des traces à Langfuse
    langfuse.flush()

# --- OBSERVATIONS SUR LA TEMPERATURE ---
# Temperature 0.1 : Réponse très structurée, classique (souvent une soupe ou un pot-au-feu), peu de variations.
# Temperature 0.7 : Bon équilibre. Le ton est plus chaleureux, les adjectifs sont plus variés.
# Temperature 1.2 : Très créatif, parfois trop. Peut inventer des noms de plats ou devenir verbeux/désordonné.