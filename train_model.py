import os
import pickle
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, roc_auc_score, recall_score, accuracy_score

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
META_PATH = os.path.join(MODEL_DIR, "model_meta.pkl")

def train_and_save():
    print("1. Téléchargement des données UCI Wine Quality...")
    df_uci = fetch_ucirepo(id=186)
    X = df_uci.data.features.copy()
    y = df_uci.data.targets.copy()

    # Encodage du type de vin
    if 'color' in df_uci.data.original.columns:
        X['is_red'] = (df_uci.data.original['color'] == 'red').astype(int)

    # Cible binaire : bon vin si note >= 7
    y_bin = (y['quality'] >= 7).astype(int)

    # Suppression des doublons stricts (traitement du data leakage)
    df_full = X.copy()
    df_full['quality'] = y['quality']
    mask_dup = df_full.duplicated()
    X_clean = X[~mask_dup].copy()
    y_clean = y_bin[~mask_dup].copy()
    print(f"   Données nettoyées : {X_clean.shape[0]} échantillons uniques.")

    # Découpage train/test stratifié
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean, test_size=0.20, stratify=y_clean, random_state=42
    )

    # Pipeline optimal : RobustScaler + SVC(C=0.5, gamma=0.2, RBF, balanced)
    print("2. Entraînement du modèle SVM (noyau RBF) optimisé...")
    svm_pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('model', SVC(
            C=0.5,
            gamma=0.2,
            kernel='rbf',
            class_weight='balanced',
            probability=True,
            random_state=42
        ))
    ])

    svm_pipeline.fit(X_train, y_train)

    # Évaluation de contrôle
    y_pred = svm_pipeline.predict(X_test)
    y_proba = svm_pipeline.predict_proba(X_test)[:, 1]
    metrics = {
        'accuracy': round(accuracy_score(y_test, y_pred), 3),
        'recall': round(recall_score(y_test, y_pred), 3),
        'f1': round(f1_score(y_test, y_pred), 3),
        'auc': round(roc_auc_score(y_test, y_proba), 3)
    }
    print(f"3. Performances sur test : {metrics}")

    # Calcul des statistiques médianes par classe
    stats_bon = X_clean[y_clean == 1].median().to_dict()
    stats_mauvais = X_clean[y_clean == 0].median().to_dict()

    # Sauvegarde avec pickle
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(svm_pipeline, f)

    meta = {
        'features': list(X_clean.columns),
        'metrics': metrics,
        'stats_bon': stats_bon,
        'stats_mauvais': stats_mauvais
    }
    with open(META_PATH, "wb") as f:
        pickle.dump(meta, f)

    print(f"4. Modèle et métadonnées sauvegardés avec succès dans {MODEL_DIR}")

if __name__ == "__main__":
    train_and_save()
