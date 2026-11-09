
# Importer Pandas pour travailler avec les données.
import pandas as pd

# Importer Path pour gérer le chemin du fichier.
from pathlib import Path

# Trouver le dossier principal DocAgent.
project_dir = Path(__file__).resolve().parents[1]

# Définir le chemin de notre dataset.
file_path = (
    project_dir
    / "data"
    / "raw"
    / "stackoverflow_high_quality.csv"
)

# Lire le fichier CSV avec Pandas.
df = pd.read_csv(file_path)

# 1. Afficher les dimensions du dataset.
print("\n=== DIMENSIONS ===")
print(df.shape)

# 2. Afficher les noms des colonnes.
print("\n=== COLONNES ===")
print(df.columns.tolist())

# 3. Afficher les cinq premières questions.
print("\n=== PREMIERES QUESTIONS ===")
print(df[["title", "tags"]].head(5).to_string(index=False))

# 4. Compter les valeurs manquantes.
print("\n=== VALEURS MANQUANTES ===")
print(df.isna().sum())

# 5. Compter les titres dupliqués.
print("\n=== DOUBLONS ===")
print(df.duplicated(subset=["title"]).sum())


# ============================================================
# ETAPE 2 : ANALYSE DES TAGS
# ============================================================

# Les tags sont séparés par le caractère "|".
# Exemple : "python|pandas|dataframe"

# Transformer chaque chaîne de tags en liste.
tags_list = df["tags"].str.split("|", regex=False)

# Placer chaque tag sur une ligne distincte.
all_tags = tags_list.explode()

# Compter les occurrences de chaque tag.
tag_counts = all_tags.value_counts()

# Afficher les 20 tags les plus fréquents.
print("\n=== TOP 20 DES TAGS ===")
print(tag_counts.head(20))

# Vérifier les tags liés à nos six technologies.
target_tags = [
    "python",
    "docker",
    "javascript",
    "typescript",
    "c#",
    ".net",
    "asp.net-core",
    "sql",
    "postgresql",
    "mysql",
    "sql-server",
    "git"
]

print("\n=== TECHNOLOGIES DE DOCAGENT ===")

for tag in target_tags:
    count = tag_counts.get(tag, 0)
    print(f"{tag} : {count} questions")

# ============================================================
# ETAPE 3 : IDENTIFIER LES TECHNOLOGIES
# ============================================================

# Définir les tags associés à nos 6 technologies.
technology_tags = {
    "python": {"python"},
    "docker": {"docker"},
    "javascript": {"javascript"},
    "dotnet": {"c#", ".net", "asp.net-core"},
    "sql": {"sql", "postgresql"},
    "git": {"git"}
}


# Fonction qui identifie les technologies d'une question.
def find_technologies(tags):

    # Séparer les tags de la question.
    # Exemple : "python|docker" devient {"python", "docker"}.
    tag_set = set(tags.split("|"))

    # Préparer une liste pour les technologies détectées.
    detected = []

    # Parcourir nos 6 catégories.
    for technology, keywords in technology_tags.items():

        # Vérifier si des tags correspondent.
        if tag_set.intersection(keywords):
            detected.append(technology)

    # Retourner les technologies trouvées.
    return detected


# Appliquer la fonction aux 20 205 questions.
df["detected_technologies"] = df["tags"].apply(
    find_technologies
)

# Compter combien de technologies sont détectées par question.
df["number_of_technologies"] = (
    df["detected_technologies"].apply(len)
)


# ============================================================
# ANALYSER LES RESULTATS
# ============================================================

print("\n=== REPARTITION DES QUESTIONS ===")

# Questions avec une seule technologie.
single = df["number_of_technologies"] == 1

# Questions avec plusieurs technologies.
multiple = df["number_of_technologies"] > 1

# Questions sans aucune technologie recherchée.
none = df["number_of_technologies"] == 0

print("Une seule technologie :", single.sum())
print("Plusieurs technologies :", multiple.sum())
print("Aucune technologie :", none.sum())


# ============================================================
# COMPTER LES QUESTIONS PAR TECHNOLOGIE
# ============================================================

# Sélectionner uniquement les questions avec un seul label.
labels = df.loc[
    single, "detected_technologies"
].apply(lambda technologies: technologies[0])

print("\n=== QUESTIONS PAR TECHNOLOGIE ===")
print(labels.value_counts())


# ============================================================
# AFFICHER LES QUESTIONS AMBIGUES
# ============================================================

print("\n=== QUESTIONS AVEC PLUSIEURS TECHNOLOGIES ===")

print(
    df.loc[
        multiple,
        ["title", "tags", "detected_technologies"]
    ].head(5).to_string(index=False)
)


# ============================================================
# ETAPE 4 : CREER UN DATASET NETTOYE
# ============================================================

# 1. Conserver uniquement les questions avec une catégorie.
clean_df = df.loc[
    single,
    ["id", "title", "tags", "detected_technologies"]
].copy()

# 2. Renommer la colonne title en question.
clean_df = clean_df.rename(
    columns={"title": "question"}
)

# 3. Récupérer l'unique technologie détectée.
clean_df["technology"] = clean_df[
    "detected_technologies"
].apply(lambda technologies: technologies[0])

# 4. Supprimer les espaces inutiles dans les questions.
clean_df["question"] = (
    clean_df["question"]
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

# 5. Supprimer les questions devenues vides.
clean_df = clean_df[
    clean_df["question"] != ""
].copy()

# 6. Détecter les questions en double, sans tenir compte
#    des majuscules et minuscules.
clean_df["normalized_question"] = (
    clean_df["question"].str.casefold()
)

duplicates = clean_df.duplicated(
    subset=["normalized_question"]
)

print("\nDoublons détectés :", duplicates.sum())

# 7. Conserver une seule occurrence par question.
clean_df = clean_df.drop_duplicates(
    subset=["normalized_question"],
    keep="first"
)

# 8. Ajouter le lien vers la question Stack Overflow.
clean_df["source_url"] = (
    "https://stackoverflow.com/questions/"
    + clean_df["id"].astype(str)
)

# 9. Conserver les colonnes utiles.
clean_df = clean_df[
    ["id", "question", "technology", "tags", "source_url"]
]

# 10. Créer le dossier processed.
output_dir = project_dir / "data" / "processed"
output_dir.mkdir(parents=True, exist_ok=True)

# 11. Enregistrer les données dans un nouveau CSV.
output_path = output_dir / "questions_clean.csv"

clean_df.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig"
)

# 12. Vérifier le résultat.
print("\n=== DATASET NETTOYE ===")
print("Nombre de questions :", len(clean_df))
print("\nRépartition par technologie :")
print(clean_df["technology"].value_counts())
print("\nPremières lignes :")
print(clean_df.head(5).to_string(index=False))
print("\nFichier créé :", output_path)

