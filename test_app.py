import os
import pickle
import numpy as np
import pandas as pd
import pytest

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "model.pkl")
META_PATH = os.path.join(CURRENT_DIR, "model_meta.pkl")


@pytest.fixture
def loaded_model_and_meta():
    """Charge le modèle sérialisé et ses métadonnées."""
    assert os.path.exists(MODEL_PATH), "Le fichier model.pkl doit exister."
    assert os.path.exists(META_PATH), "Le fichier model_meta.pkl doit exister."
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    return model, meta


def test_model_pipeline_structure(loaded_model_and_meta):
    """Vérifie que le pipeline contient bien le RobustScaler et le SVC."""
    model, meta = loaded_model_and_meta
    assert hasattr(model, "predict"), "Le modèle doit posséder la méthode predict."
    assert hasattr(model, "predict_proba"), "Le modèle doit posséder predict_proba."
    step_names = [name for name, _ in model.steps]
    assert "scaler" in step_names, "Le pipeline doit inclure une étape 'scaler'."
    assert "model" in step_names, "Le pipeline doit inclure une étape 'model'."


def test_expected_features_count(loaded_model_and_meta):
    """Vérifie que les 12 descripteurs exacts sont attendus par le modèle."""
    _, meta = loaded_model_and_meta
    expected_features = [
        'fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
        'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
        'pH', 'sulphates', 'alcohol', 'is_red'
    ]
    assert meta['features'] == expected_features, "Les variables doivent correspondre aux 12 descripteurs retenus."


def test_prediction_output_format(loaded_model_and_meta):
    """Vérifie le format des prédictions et des probabilités retournées."""
    model, meta = loaded_model_and_meta
    sample = pd.DataFrame([{
        'fixed_acidity': 7.0,
        'volatile_acidity': 0.25,
        'citric_acid': 0.30,
        'residual_sugar': 2.5,
        'chlorides': 0.040,
        'free_sulfur_dioxide': 30.0,
        'total_sulfur_dioxide': 120.0,
        'density': 0.9920,
        'pH': 3.20,
        'sulphates': 0.50,
        'alcohol': 12.0,
        'is_red': 0
    }])[meta['features']]

    pred = model.predict(sample)
    proba = model.predict_proba(sample)

    assert pred[0] in [0, 1], "La prédiction binaire doit valoir 0 ou 1."
    assert proba.shape == (1, 2), "Les probabilités doivent avoir 2 colonnes (classes 0 et 1)."
    assert np.isclose(proba.sum(), 1.0), "La somme des probabilités doit être égale à 1."


def test_consistency_grand_cru_vs_mediocre(loaded_model_and_meta):
    """Vérifie la cohérence œnologique : un profil d'excellence a une probabilité > profil médiocre."""
    model, meta = loaded_model_and_meta

    grand_cru = pd.DataFrame([{
        'fixed_acidity': 7.2,
        'volatile_acidity': 0.20,      # Très faible acidité volatile
        'citric_acid': 0.35,
        'residual_sugar': 2.0,
        'chlorides': 0.035,
        'free_sulfur_dioxide': 35.0,
        'total_sulfur_dioxide': 110.0,
        'density': 0.9910,             # Faible densité
        'pH': 3.25,
        'sulphates': 0.55,
        'alcohol': 13.0,               # Degré d'alcool élevé
        'is_red': 0
    }])[meta['features']]

    vin_mediocre = pd.DataFrame([{
        'fixed_acidity': 7.5,
        'volatile_acidity': 0.70,      # Fort défaut acétique (vinaigre)
        'citric_acid': 0.10,
        'residual_sugar': 2.0,
        'chlorides': 0.090,
        'free_sulfur_dioxide': 10.0,
        'total_sulfur_dioxide': 60.0,
        'density': 0.9990,             # Forte densité
        'pH': 3.40,
        'sulphates': 0.40,
        'alcohol': 9.2,                # Faible degré d'alcool
        'is_red': 1
    }])[meta['features']]

    proba_gc = model.predict_proba(grand_cru)[0][1]
    proba_med = model.predict_proba(vin_mediocre)[0][1]

    assert proba_gc > proba_med, "Le grand cru doit obtenir une probabilité de bon vin nettement supérieure au vin médiocre."
