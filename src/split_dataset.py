
# ============================================================
# DOCAGENT - SEPARATION TRAIN / TEST
# ============================================================

# Importer Pandas pour lire les données.
import pandas as pd

# Importer la fonction de séparation Scikit-learn.
from sklearn.model_selection import train_test_split

# Importer Path pour gérer les chemins.
from pathlib import Path


# ============================================================
# 1. CHARGER LE DATASET NETTOYE
# ============================================================

# Retrouver le dossier principal de DocAgent.
project_dir = Path(__file__).resolve().parents[1]

# Définir le chemin de notre CSV nettoyé.
file_path = project_dir / "data" / "processed" / "questions_clean.csv"

# Charger les questions et leurs technologies.
df = pd.read_csv(file_path)

print("\n=== DATASET INITIAL ===")
print("Nombre de questions :", len(df))

# Vérifier les colonnes nécessaires.
required_columns = {"id", "question", "technology"}

if not required_columns.issubset(df.columns):
    raise ValueError("Colonnes nécessaires manquantes.")

# Vérifier les valeurs manquantes.
if df[["question", "technology"]].isna().any().any():
    raise ValueError("Des questions ou des labels sont manquants.")

# Vérifier qu'il n'y a pas de questions vides.
if df["question"].str.strip().eq("").any():
    raise ValueError("Certaines questions sont vides.")

# Vérifier les doublons de questions.
normalized = df["question"].str.strip().str.casefold()

if normalized.duplicated().any():
    raise ValueError("Des questions dupliquées ont été détectées.")


# ============================================================
# 2. AFFICHER LA REPARTITION DES CLASSES
# ============================================================

print("\n=== QUESTIONS PAR TECHNOLOGIE ===")
print(df["technology"].value_counts())


# ============================================================
# 3. SEPARER LES DONNEES
# ============================================================

# 80 % pour apprendre, 20 % pour tester.
train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["technology"]
)

# random_state=42 rend le tirage reproductible.
# stratify conserve approximativement les proportions des classes.


# ============================================================
# 4. VERIFIER LA SEPARATION
# ============================================================

print("\n=== RESULTATS DE LA SEPARATION ===")

print("Questions entraînement :", len(train_df))
print("Questions test :", len(test_df))

# Vérifier qu'aucune question n'est présente dans les deux.
common_ids = set(train_df["id"]) & set(test_df["id"])

print("Questions communes :", len(common_ids))

if common_ids:
    raise ValueError("Des questions sont présentes dans les deux jeux.")

print("\n=== REPARTITION TRAIN ===")
print(train_df["technology"].value_counts())

print("\n=== REPARTITION TEST ===")
print(test_df["technology"].value_counts())


# ============================================================
# 5. ENREGISTRER LES FICHIERS
# ============================================================

# Créer un dossier spécifique pour les deux fichiers.
output_dir = project_dir / "data" / "splits"

output_dir.mkdir(parents=True, exist_ok=True)

# Enregistrer le jeu d'entraînement.
train_df.to_csv(
    output_dir / "train.csv",
    index=False,
    encoding="utf-8-sig"
)

# Enregistrer le jeu de test.
test_df.to_csv(
    output_dir / "test.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nFichiers créés avec succès !")
print("train.csv et test.csv")
