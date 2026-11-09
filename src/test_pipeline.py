
# ============================================================
# DOCAGENT - INTEGRATION DU MODELE ML AVEC LES OUTILS PYTHON
# ============================================================

# Importer Joblib pour charger notre modèle ML sauvegardé.
from joblib import load

# Importer Path pour retrouver les fichiers du projet.
from pathlib import Path

# Importer notre liste blanche des sites officiels.
from src.config import OFFICIAL_SITES

# Importer les trois outils développés précédemment.
from src.tools import search_docs, fetch_page, extract_section


# ============================================================
# 1. CHARGER NOTRE MODELE ML
# ============================================================

# Récupérer le dossier principal DocAgent.
project_dir = Path(__file__).resolve().parents[1]

# Construire le chemin vers notre modèle entraîné.
model_path = project_dir / "models" / "docagent_router.joblib"

# Charger uniquement notre modèle local, créé par nous-mêmes.
model = load(model_path)


# ============================================================
# 2. CREER NOTRE PREMIER PIPELINE DOCAGENT
# ============================================================

def run_pipeline(question):

    # Vérifier que la question n'est pas vide.
    question = question.strip()

    if not question:
        print("Question vide.")
        return

    # ETAPE A : IDENTIFIER LA TECHNOLOGIE

    # Le modèle ML transforme automatiquement la question
    # avec TF-IDF et prédit sa technologie.
    technology = model.predict([question])[0]

    print("\n=== PREDICTION ML ===")
    print("Question :", question)
    print("Technologie prédite :", technology)

    # ETAPE B : VERIFIER LA TECHNOLOGIE

    # Vérifier qu'elle correspond à notre liste blanche.
    if technology not in OFFICIAL_SITES:
        print("Technologie non autorisée.")
        return

    # Récupérer le domaine officiel correspondant.
    domain = OFFICIAL_SITES[technology]

    print("Site officiel :", domain)

    # ETAPE C : RECHERCHER LES DOCUMENTATIONS

    print("\n=== RECHERCHE DOCUMENTATION ===")

    # Utiliser la question originale complète.
    pages = search_docs(question, technology)

    if not pages:
        print("Aucune page officielle trouvée par DDGS.")
        return

    # Préparer une liste pour les passages pertinents.
    all_passages = []

    # ETAPE D : TELECHARGER LES PAGES

    # Parcourir au maximum trois résultats officiels.
    for page in pages[:3]:

        print("\nPage :", page["url"])

        # Télécharger le contenu HTML et extraire le texte.
        document = fetch_page(page["url"])

        # Passer à la page suivante si le téléchargement échoue.
        if document is None:
            print("Impossible de lire cette page.")
            continue

        # ETAPE E : SELECTIONNER LES PASSAGES

        passages = extract_section(
            document,
            question,
            max_passages=3
        )

        # Ajouter les passages à notre liste générale.
        all_passages.extend(passages)

    # ETAPE F : CLASSER LES PASSAGES

    # Classer les résultats par score décroissant.
    all_passages.sort(
        key=lambda passage: passage["score"],
        reverse=True
    )

    print("\n=== PASSAGES PERTINENTS ===")

    if not all_passages:
        print("Aucun passage pertinent récupéré.")
        return

    # Afficher les trois meilleurs passages.
    for index, passage in enumerate(
        all_passages[:3],
        start=1
    ):

        print(f"\nPassage {index}")
        print("Score :", passage["score"])
        print("Texte :", passage["text"][:600])
        print("Source :", passage["url"])


# ============================================================
# 3. TESTER LE PIPELINE
# ============================================================

if __name__ == "__main__":

    # Demander une question à l'utilisateur.
    user_question = input(
        "\nPose une question informatique : "
    )

    # Exécuter notre pipeline.
    run_pipeline(user_question)
