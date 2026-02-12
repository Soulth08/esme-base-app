import os
from dotenv import load_dotenv
from langfuse import observe, get_client, propagate_attributes
import litellm


# Chargement des variables d'environnement
load_dotenv()

@observe(name="ask_chef_call")
def ask_chef(question: str, temperature: float = 0.7) -> str:

    # tags
    with propagate_attributes(tags=["COLPIN / MORETTI", "1.3"]):

        system_prompt = (
            "Tu es ChefBot, un chef cuisinier français renommé, "
            "spécialisé dans la cuisine de saison et les produits locaux."
        )

        response = litellm.completion(
            model="groq/llama-3.1-8b-instant", # on utilise ce modèle car il est plus petit, donc rate limit plus bas
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            temperature=temperature,
            api_key=os.getenv("GROQ_API_KEY")
        )

        return response.choices[0].message.content

if __name__ == "__main__":
    
    #langfuse client
    langfuse = get_client()
    user_query = "En tant que chef cuisinier français spécialisé en cuisine de saison, peux-tu me suggérer la recette d'un plat ?"
    
    # 1.3 températures
    temperatures = [0.1, 0.7, 1.2, 2.0]
    
    print(f"👤 Question: {user_query}\n")

    for temp in temperatures:
        print(f"température = {temp}")
        res = ask_chef(user_query, temperature=temp)
        print(f"ChefBot: {res}\n")

    # Envoi final des traces à Langfuse
    langfuse.flush()

# --- OBSERVATIONS SUR LA TEMPERATURE ---
# Temperature 0.1 : La réponse est structurée et claire. par contre c'est très souvent la même recette !
# Temperature 0.7 : Parfois n'est pas cohérent. On a par exemple eu des "fruits de saison" en hiver, ce qui n'ets pas logique.
# Temperature 1.2 : Très créatif, parfois trop. Peut inventer des noms de plats (escargots en croûte de bacon XD) ou devenir verbeux/désordonné.
# Temperature 2.0 : Nous avons essayé avec une température plus élevée juste par curiosité. les phrases sont totalement désordonnées avec des fautes de frappe partout (Dans les épis des deux poire des deux couteaux dans un bon rétrique à coute.).