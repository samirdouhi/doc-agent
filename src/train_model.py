
# ============================================================
# DOCAGENT - PREMIER ENTRAINEMENT MACHINE LEARNING
# ============================================================

# Importer Pandas pour lire notre fichier CSV.
import pandas as pd

# Importer Path pour gérer les chemins.
from pathlib import Path

# Importer l'outil de séparation des données.
from sklearn.model_selection import train_test_split

# Importer TF-IDF pour transformer les textes en nombres.
from sklearn.feature_extraction.text import TfidfVectorizer

# Importer notre algorithme de classification.
from sklearn.linear_model import LogisticRegression

# Importer Pipeline pour connecter TF-IDF au modèle.
from sklearn.pipeline import Pipeline

# Importer les métriques pour évaluer les prédictions.
from sklearn.metrics import classification_report


# ============================================================
# 1. CHARGER LES DONNEES D'ENTRAINEMENT
# ============================================================

project_dir = Path(__file__).resolve().parents[1]

train_path = project_dir / "data" / "splits" / "train.csv"

df = pd.read_csv(train_path)

# X représente les questions.
X = df["question"]

# y représente les bonnes réponses : les technologies.
y = df["technology"]

print("\n=== DONNEES CHARGEES ===")
print("Nombre de questions :", len(df))


# ============================================================
# 2. CREER UN ENSEMBLE DE VALIDATION
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n=== REPARTITION ===")
print("Entraînement :", len(X_train))
print("Validation :", len(X_val))


# ============================================================
# 3. CREER NOTRE MODELE MACHINE LEARNING
# ============================================================

model = Pipeline([

    # TF-IDF transforme les mots en nombres.
    (
        "tfidf",
        TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2
        )
    ),

    # Régression logistique apprend à prédire la technologie.
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    )

])


# ============================================================
# 4. ENTRAINER LE MODELE
# ============================================================

print("\n=== ENTRAINEMENT ===")

# Le modèle apprend uniquement à partir de X_train et y_train.
model.fit(X_train, y_train)

print("Entraînement terminé !")


# ============================================================
# 5. EFFECTUER DES PREDICTIONS
# ============================================================

# Prédire les technologies des questions de validation.
predictions = model.predict(X_val)


# ============================================================
# 6. EVALUER LE MODELE
# ============================================================

print("\n=== RAPPORT DE CLASSIFICATION ===")

# Comparer les prédictions avec les bonnes réponses.
print(
    classification_report(
        y_val,
        predictions,
        zero_division=0
    )
)

# ============================================================
# ETAPE 5.6 : CORRIGER LE DESEQUILIBRE DES CLASSES
# ============================================================

# Importer clone pour créer un deuxième modèle indépendant.
from sklearn.base import clone

# Importer f1_score pour comparer les performances.
from sklearn.metrics import f1_score


# ============================================================
# 1. CREER UN MODELE AVEC EQUILIBRAGE
# ============================================================

print("\n=== MODELE AVEC CLASS_WEIGHT BALANCED ===")

# Copier la structure de notre premier modèle.
balanced_model = clone(model)

# Activer la pondération automatique des classes.
# Les classes minoritaires recevront davantage de poids.
balanced_model.set_params(
    classifier__class_weight="balanced"
)


# ============================================================
# 2. ENTRAINER LE DEUXIEME MODELE
# ============================================================

# Utiliser exactement les mêmes données d'entraînement.
balanced_model.fit(X_train, y_train)

print("Entraînement du modèle équilibré terminé !")


# ============================================================
# 3. EVALUER LE NOUVEAU MODELE
# ============================================================

# Prédire les technologies des questions de validation.
balanced_predictions = balanced_model.predict(X_val)

# Afficher les performances.
print("\n=== RAPPORT DU MODELE EQUILIBRE ===")

print(
    classification_report(
        y_val,
        balanced_predictions,
        zero_division=0
    )
)


# ============================================================
# 4. COMPARER LES DEUX MODELES
# ============================================================

# Calculer le F1-score macro de notre premier modèle.
baseline_f1 = f1_score(
    y_val,
    predictions,
    average="macro"
)

# Calculer le F1-score macro du modèle équilibré.
balanced_f1 = f1_score(
    y_val,
    balanced_predictions,
    average="macro"
)

print("\n=== COMPARAISON ===")

print(
    "F1 macro modèle initial :",
    round(baseline_f1, 3)
)

print(
    "F1 macro modèle équilibré :",
    round(balanced_f1, 3)
)

# ============================================================
# ETAPE 5.7 : EVALUATION FINALE SUR LE JEU DE TEST
# ============================================================

# ============================================================
# 1. CHARGER LES DONNEES DE TEST
# ============================================================

test_path = project_dir / "data" / "splits" / "test.csv"

test_df = pd.read_csv(test_path)

# Récupérer les questions et leurs vrais labels.
X_test = test_df["question"]
y_test = test_df["technology"]

print("\n=== DONNEES DE TEST ===")
print("Nombre de questions :", len(test_df))


# ============================================================
# 2. ENTRAINER LE MODELE FINAL
# ============================================================

# Créer une copie neuve du modèle équilibré.
final_model = clone(balanced_model)

# Entraîner sur les 4849 questions du train.csv.
# Ne jamais utiliser test.csv pour entraîner le modèle.
final_model.fit(X, y)

print("\nModèle final entraîné sur toutes les données train.")


# ============================================================
# 3. EFFECTUER LES PREDICTIONS
# ============================================================

# Prédire les technologies des questions jamais vues.
test_predictions = final_model.predict(X_test)


# ============================================================
# 4. EVALUER LES PREDICTIONS
# ============================================================

print("\n=== RAPPORT FINAL SUR LE TEST ===")

print(
    classification_report(
        y_test,
        test_predictions,
        zero_division=0
    )
)

# Calculer le F1-score macro final.
final_f1 = f1_score(
    y_test,
    test_predictions,
    average="macro"
)

print("\nF1-score macro final :", round(final_f1, 3))


# ============================================================
# ETAPE 5.8 : SAUVEGARDER LE MODELE ENTRAINE
# ============================================================

# Importer joblib pour enregistrer notre modèle.
from joblib import dump

# Définir le dossier où sera stocké le modèle.
model_dir = project_dir / "models"

# Créer le dossier s'il n'existe pas.
model_dir.mkdir(parents=True, exist_ok=True)

# Définir le chemin du fichier.
model_path = model_dir / "docagent_router.joblib"

# Enregistrer notre Pipeline complet :
# TF-IDF + Régression logistique équilibrée.
dump(final_model, model_path)

print("\n=== SAUVEGARDE DU MODELE ===")
print("Modèle sauvegardé :", model_path)
