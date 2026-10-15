import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Qualité du vin", layout="wide")

FEATURES = ["fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
            "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
            "pH", "sulphates", "alcohol", "is_red"]
NOMS = {"fixed_acidity": "Acidité fixe", "volatile_acidity": "Acidité volatile",
        "citric_acid": "Acide citrique", "residual_sugar": "Sucre résiduel",
        "chlorides": "Chlorures", "free_sulfur_dioxide": "SO₂ libre",
        "total_sulfur_dioxide": "SO₂ total", "density": "Densité", "pH": "pH",
        "sulphates": "Sulfates", "alcohol": "Alcool", "is_red": "Vin rouge"}
MESURES = FEATURES[:-1]
MODELES = ["Régression logistique", "k-NN", "Arbre de décision", "Naive Bayes", "SVM"]

# Meilleurs paramètres trouvés dans le notebook (GridSearch / RandomizedSearch, CV 5 plis, F1)
GRILLES = pd.DataFrame({
    "Recherche": ["Grid", "Grid", "Grid", "Grid", "Random (15 essais)"],
    "Valeurs testées": ["C : 0,001 → 100",
                        "k : 3 → 21, poids, p = 1 ou 2",
                        "profondeur 3 → illimitée, 1 → 20 par feuille, gini / entropy",
                        "var_smoothing : 1e-11 → 1e-5",
                        "C : 0,1 → 10, gamma : 0,01 → 1"],
    "Retenu": ["C = 0,01", "k = 5, Manhattan, poids par distance",
               "entropy, profondeur 3, 20 par feuille", "1e-11", "C = 0,5, gamma = 0,2"]},
    index=MODELES)

CSV_LOCAL = Path(__file__).parent / "wine_quality.csv"
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def fr(x, n=3):
    return f"{x:.{n}f}".replace(".", ",")


def montrer(fig):
    st.pyplot(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Données
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Chargement des données…")
def charger():
    if CSV_LOCAL.exists():
        df = pd.read_csv(CSV_LOCAL)
    else:
        from ucimlrepo import fetch_ucirepo
        df = fetch_ucirepo(id=186).data.original
        try:
            df.to_csv(CSV_LOCAL, index=False)  # copie locale : plus besoin d'internet ensuite
        except OSError:
            pass
    df["is_red"] = (df["color"] == "red").astype(int)
    return df[FEATURES].reset_index(drop=True), df["quality"].reset_index(drop=True)


@st.cache_data
def preparer():
    X, quality = charger()
    y_bin = (quality >= 7).astype(int)
    doublons = X.assign(quality=quality).duplicated()
    Xc, yc = X[~doublons], y_bin[~doublons]
    X_train, X_test, y_train, y_test = train_test_split(
        Xc, yc, test_size=0.2, stratify=yc, random_state=42)
    return dict(X=X, quality=quality, y_bin=y_bin, doublons=doublons,
                X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


# ---------------------------------------------------------------------------
# Modèles
# ---------------------------------------------------------------------------
def modele(nom, optimise=False):
    if optimise:
        m = {"Régression logistique": LogisticRegression(C=0.01, class_weight="balanced", max_iter=2000),
             "k-NN": KNeighborsClassifier(n_neighbors=5, p=1, weights="distance"),
             "Arbre de décision": DecisionTreeClassifier(criterion="entropy", max_depth=3, min_samples_leaf=20,
                                                         class_weight="balanced", random_state=42),
             "Naive Bayes": GaussianNB(var_smoothing=1e-11),
             "SVM": SVC(C=0.5, gamma=0.2, class_weight="balanced", random_state=42)}[nom]
    else:
        m = {"Régression logistique": LogisticRegression(class_weight="balanced", max_iter=2000),
             "k-NN": KNeighborsClassifier(),
             "Arbre de décision": DecisionTreeClassifier(class_weight="balanced", random_state=42),
             "Naive Bayes": GaussianNB(),
             "SVM": SVC(class_weight="balanced", random_state=42)}[nom]
    return Pipeline([("scaler", RobustScaler()), ("model", m)])


@st.cache_resource(show_spinner="Entraînement des modèles…")
def entrainer(optimise):
    d = preparer()
    return {nom: modele(nom, optimise).fit(d["X_train"], d["y_train"]) for nom in MODELES}


def score(m, X):
    if hasattr(m, "predict_proba"):
        return m.predict_proba(X)[:, 1]
    return m.decision_function(X)


def metriques(y, pred, s=None):
    r = {"Accuracy": accuracy_score(y, pred), "Précision": precision_score(y, pred, zero_division=0),
         "Rappel": recall_score(y, pred), "F1": f1_score(y, pred)}
    if s is not None:
        r["AUC"] = roc_auc_score(y, s)
    return r


@st.cache_data(show_spinner=False)
def tableau_test(optimise):
    d = preparer()
    mods = entrainer(optimise)
    return pd.DataFrame({nom: metriques(d["y_test"], m.predict(d["X_test"]), score(m, d["X_test"]))
                         for nom, m in mods.items()}).T


# ---------------------------------------------------------------------------
# Calculs longs (mis en cache)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Test k-NN avec / sans doublons…")
def effet_doublons():
    d = preparer()
    X, y = d["X"], d["y_bin"]
    out = {}
    for nom, masque in [("Avec doublons", np.ones(len(X), bool)), ("Sans doublons", ~d["doublons"].values)]:
        Xa, Xb, ya, yb = train_test_split(X[masque], y[masque], test_size=0.2,
                                          stratify=y[masque], random_state=42)
        knn = Pipeline([("s", RobustScaler()), ("m", KNeighborsClassifier(5, weights="distance"))]).fit(Xa, ya)
        out[nom] = f1_score(yb, knn.predict(Xb))
    return out


@st.cache_data(show_spinner="Validation croisée des deux scalers…")
def comparer_scalers():
    d = preparer()
    lignes = {}
    for nom in MODELES:
        m = modele(nom).named_steps["model"]
        r = cross_val_score(Pipeline([("s", RobustScaler()), ("m", m)]), d["X_train"], d["y_train"],
                            cv=CV, scoring="f1", n_jobs=-1)
        s = cross_val_score(Pipeline([("s", StandardScaler()), ("m", m)]), d["X_train"], d["y_train"],
                            cv=CV, scoring="f1", n_jobs=-1)
        lignes[nom] = {"F1 RobustScaler": f"{fr(r.mean())} ± {fr(r.std())}",
                       "F1 StandardScaler": f"{fr(s.mean())} ± {fr(s.std())}"}
    return pd.DataFrame(lignes).T


@st.cache_data(show_spinner="Calcul des critères de sélection…")
def criteres_selection():
    d = preparer()
    Xtr, ytr = d["X_train"], d["y_train"]
    lr = LogisticRegression(class_weight="balanced", max_iter=2000).fit(StandardScaler().fit_transform(Xtr), ytr)
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1).fit(Xtr, ytr)
    return pd.DataFrame({"Correlation": Xtr.corrwith(ytr), "Coef_reg_logistique": lr.coef_[0],
                         "Importance_RF": rf.feature_importances_}, index=Xtr.columns)


@st.cache_data(show_spinner="Validation croisée : toutes les variables / 8 variables / ACP…")
def comparer_jeux_variables():
    d = preparer()
    Xtr, ytr = d["X_train"], d["y_train"]
    top8 = list(criteres_selection()["Importance_RF"].sort_values(ascending=False).index[:8])
    lignes = {}
    for nom in MODELES:
        m = modele(nom).named_steps["model"]
        ligne = {}
        for jeu, pipe, cols in [
            ("Toutes les variables", Pipeline([("s", RobustScaler()), ("m", m)]), FEATURES),
            ("Sélection RF (8)", Pipeline([("s", RobustScaler()), ("m", m)]), top8),
            ("ACP (90 %)", Pipeline([("s", RobustScaler()), ("acp", PCA(0.90)), ("m", m)]), FEATURES)]:
            sc = cross_val_score(pipe, Xtr[cols], ytr, cv=CV, scoring="f1", n_jobs=-1)
            ligne[jeu] = f"{fr(sc.mean())} ± {fr(sc.std())}"
        lignes[nom] = ligne
    return pd.DataFrame(lignes).T, top8


@st.cache_data(show_spinner="Train / validation croisée / test…")
def biais_variance():
    d = preparer()
    mods = entrainer(True)
    lignes = {}
    for nom, m in mods.items():
        cv = cross_val_score(modele(nom, True), d["X_train"], d["y_train"], cv=CV, scoring="f1", n_jobs=-1)
        lignes[nom] = {"F1 train": f1_score(d["y_train"], m.predict(d["X_train"])),
                       "F1 validation croisée": cv.mean(),
                       "F1 test": f1_score(d["y_test"], m.predict(d["X_test"]))}
    t = pd.DataFrame(lignes).T
    t["Écart train - CV"] = t["F1 train"] - t["F1 validation croisée"]
    return t


# ---------------------------------------------------------------------------
# Graphiques du notebook (mêmes réglages et mêmes couleurs)
# ---------------------------------------------------------------------------
def heatmap_confusion(ax, cm, titre):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Mauvais", "Bon"], yticklabels=["Mauvais", "Bon"])
    ax.set_title(titre)
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Réel")


def fig_confusions(mods, titre):
    d = preparer()
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for ax, (nom, m) in zip(axes, mods.items()):
        heatmap_confusion(ax, confusion_matrix(d["y_test"], m.predict(d["X_test"])), nom)
    plt.suptitle("Matrices de confusion - " + titre)
    plt.tight_layout()
    return fig


def fig_roc(mods, titre):
    d = preparer()
    fig = plt.figure(figsize=(7, 6))
    for nom, m in mods.items():
        s = score(m, d["X_test"])
        fpr, tpr, _ = roc_curve(d["y_test"], s)
        plt.plot(fpr, tpr, label=f"{nom} (AUC = {roc_auc_score(d['y_test'], s):.2f})")
    plt.plot([0, 1], [0, 1], "r--", label="Modèle aléatoire (AUC = 0.50)")
    plt.xlabel("Taux de Faux Positifs (FPR)")
    plt.ylabel("Taux de Vrais Positifs (TPR)")
    plt.title("Courbes ROC - " + titre)
    plt.legend(loc=4)
    plt.grid(True, linestyle=":", alpha=0.6)
    return fig


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------
PAGES = ["Le sujet", "Les données", "Distributions et valeurs aberrantes", "Corrélations", "ACP",
         "Sélection de variables", "Les cinq modèles", "Résultats", "Biais / variance",
         "Conclusion", "Prédire un vin"]

if "p" not in st.session_state:
    st.session_state.p = 0


def aller(i):
    st.session_state.p = i


with st.sidebar:
    st.markdown("**Qualité du vin**")
    st.radio("Plan", list(range(len(PAGES))), key="p",
             format_func=lambda i: f"{i + 1:>2}. {PAGES[i]}", label_visibility="collapsed")
    st.caption("Anissa Brignone, Perrine Grelet")

try:
    D = preparer()
except Exception as e:
    st.error("Impossible de récupérer les données avec ucimlrepo. Vérifie ta connexion internet "
             "et que le paquet est installé (pip install ucimlrepo), puis relance.")
    st.caption(f"Erreur : {e}")
    st.stop()

X, quality, y_bin = D["X"], D["quality"], D["y_bin"]
page = PAGES[st.session_state.p]

# ---------------------------------------------------------------------------
if page == "Le sujet":
    st.title("Prédiction de la qualité du vin à partir de ses caractéristiques ")
    n_vins = f"{len(X):,}".replace(",", " ")
    c1, c2 = st.columns([1, 1.2])
    with c1:
        st.markdown(f"""
Notre base de données se compose de **{n_vins} vins** (rouges et blancs, nord du Portugal).
Pour chaque vin on a **11 mesures** (acidité, sucre, alcool...)
et une **note de 0 à 10** donnée par des dégustateurs.

Le sujet nous demande d'en faire une classification binaire : un vin est **bon si sa note est ≥ 7**.

Ce qui rend le problème difficile, on le voit tout de suite sur le graphique : les notes 5 et 6
représentent {fr(100 * quality.isin([5, 6]).mean(), 1)} % des vins, et les notes extrêmes sont très rares.
""")
    with c2:
        counts = quality.value_counts().sort_index()
        fig = plt.figure()
        bars = plt.bar(counts.index, counts.values, width=0.6)
        plt.bar_label(bars, padding=3)
        plt.xlabel("Score de qualité")
        plt.ylabel("Effectif")
        plt.title("Répartition des scores de qualité")
        montrer(fig)

# ---------------------------------------------------------------------------
elif page == "Les données":
    st.title("Les données")
    st.write(f"{len(X)} vins, 12 variables (les 11 mesures + une variable rouge/blanc qu'on a ajoutée), "
             f"**{int(X.isna().sum().sum())} valeur manquante**. Toutes les variables sont numériques, "
             "il n'y a rien à encoder à part la couleur.")
    with st.expander("Statistiques descriptives"):
        st.dataframe(X.describe().T.round(3))

    st.subheader("Passage en binaire : bon vin si note ≥ 7")
    c1, c2 = st.columns([1, 1.2])
    with c1:
        counts = y_bin.value_counts().sort_index()
        fig = plt.figure(figsize=(5, 4))
        bars = plt.bar(["Mauvais (<7)", "Bon (>=7)"], counts.values, color=["salmon", "seagreen"])
        plt.bar_label(bars, padding=3)
        plt.ylabel("Effectif")
        plt.title("Répartition des classes (binaire)")
        montrer(fig)
    with c2:
        st.markdown(f"""
Environ **{fr(100 * (1 - y_bin.mean()), 0)} % de mauvais vins pour {fr(100 * y_bin.mean(), 0)} % de bons**.

Le jeu est déséquilibré : un modèle qui répond toujours « mauvais » aurait déjà une bonne accuracy
sans rien avoir appris. On jugera donc les modèles sur le F1, le rappel et l'AUC,
et on utilisera `class_weight='balanced'` quand c'est possible.
""")

    st.subheader("Le piège des doublons")
    nd = int(D["doublons"].sum())
    e = effet_doublons()
    st.markdown(f"""
**{nd} lignes sont en double** ({fr(100 * nd / len(X), 1)} % du jeu) : mêmes mesures, même note.
Si on les garde, un vin peut se retrouver à la fois dans le train et dans le test, et un modèle qui « mémorise »
comme le k-NN retrouve la bonne réponse sans rien avoir appris.

On l'a vérifié : même k-NN, même découpage, le F1 passe de **{fr(e['Avec doublons'])}** avec les doublons à
**{fr(e['Sans doublons'])}** sans. On les supprime donc **avant** de séparer train et test.
""")

    st.subheader("Découpage")
    st.write(f"Il reste {len(X) - nd} vins. Train : **{len(D['X_train'])}** (80 %), test : **{len(D['X_test'])}** (20 %), "
             f"découpage stratifié donc {fr(100 * D['y_train'].mean(), 1)} % de bons vins des deux côtés. "
             "Le test n'est plus touché jusqu'à l'évaluation finale : la mise à l'échelle et la sélection de "
             "variables sont apprises sur le train seulement.")

# ---------------------------------------------------------------------------
elif page == "Distributions et valeurs aberrantes":
    st.title("Distributions et valeurs aberrantes")

    st.subheader("Histogrammes")
    axes = X.hist(bins=30, figsize=(16, 10), color="steelblue")
    fig = axes.flat[0].get_figure()
    plt.tight_layout()
    montrer(fig)
    asym = X[MESURES].skew().round(2).sort_values(ascending=False)
    st.write("Plusieurs variables sont fortement asymétriques à droite : "
             + ", ".join(f"{c} ({fr(v, 2)})" for c, v in asym.head(3).items())
             + ". pH, density et alcohol le sont peu, et is_red est binaire.")

    st.subheader("Boxplots")
    onglet1, onglet2 = st.tabs(["Toutes les données", "Selon la classe"])
    cols = X.columns
    n_cols = 4
    n_rows = int(np.ceil(len(cols) / n_cols))
    with onglet1:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3.5 * n_rows))
        for ax, col in zip(axes.flat, cols):
            sns.boxplot(y=X[col], ax=ax, color="steelblue")
            ax.set_title(col)
            ax.set_ylabel("")
        for ax in axes.flat[len(cols):]:
            ax.set_visible(False)
        plt.suptitle("Boxplots des variables physico-chimiques", y=1.02, fontsize=14)
        plt.tight_layout()
        montrer(fig)
    with onglet2:
        data = X.copy()
        data["bon_vin"] = y_bin.values
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3.5 * n_rows))
        for ax, col in zip(axes.flat, cols):
            sns.boxplot(x="bon_vin", y=col, data=data, ax=ax, palette=["salmon", "seagreen"])
            ax.set_title(col)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_xticklabels(["Mauvais", "Bon"])
        for ax in axes.flat[len(cols):]:
            ax.set_visible(False)
        plt.suptitle("Variables selon la classe (bon vin = qualité ≥ 7)", y=1.02, fontsize=14)
        plt.tight_layout()
        montrer(fig)
        st.write("Les bons vins ont une teneur en alcool nettement plus élevée, et une acidité volatile "
                 "et une densité plus faibles.")

    st.subheader("Combien de valeurs extrêmes ?")
    q1, q3 = X[MESURES].quantile(0.25), X[MESURES].quantile(0.75)
    iqr = q3 - q1
    hors = ((X[MESURES] < q1 - 1.5 * iqr) | (X[MESURES] > q3 + 1.5 * iqr))
    t = pd.DataFrame({"Nb hors bornes": hors.sum(), "%": (100 * hors.mean()).round(1)})
    t = t.sort_values("Nb hors bornes", ascending=False)
    c1, c2 = st.columns([1, 1.3])
    with c1:
        st.dataframe(t)
        st.caption("Règle de l'écart interquartile : hors de [Q1 − 1,5·IQR ; Q3 + 1,5·IQR].")
    with c2:
        st.markdown("""
Ces valeurs sont extrêmes mais chimiquement possibles, rien n'indique une erreur de saisie.
Les supprimer ferait aussi perdre des bons vins, qui sont déjà rares. **On les garde**, et on utilise un
`RobustScaler` (médiane et écart interquartile) plutôt qu'un `StandardScaler`, pour qu'elles pèsent moins.

Est-ce que ça change vraiment quelque chose ? Comparaison en validation croisée (F1, 5 plis) :
""")
        st.table(comparer_scalers())
        st.caption("Les écarts sont plus petits que l'écart-type entre plis : aucune différence mesurable. "
                   "On garde le RobustScaler pour la raison théorique, sans dire qu'il améliore les scores.")

# ---------------------------------------------------------------------------
elif page == "Corrélations":
    st.title("Corrélations")
    c1, c2 = st.columns([1, 1])
    with c1:
        corr_cible = X.corrwith(quality).sort_values()
        ax = corr_cible.plot(kind="barh", figsize=(7, 5), color="steelblue")
        plt.title("Corrélation avec le score de qualité")
        montrer(ax.get_figure())
    with c2:
        st.markdown(f"""
L'alcool est de loin la variable la plus corrélée à la qualité (**{fr(corr_cible['alcohol'], 2)}**),
suivi de la densité ({fr(corr_cible['density'], 2)}), de l'acidité volatile ({fr(corr_cible['volatile_acidity'], 2)})
et des chlorures ({fr(corr_cible['chlorides'], 2)}).

Toutes les autres sont sous 0,12 en valeur absolue. Les liens linéaires avec la cible sont donc faibles à modérés :
on s'attend à des performances modestes.
""")

    st.subheader("Matrice de corrélation")
    correlation_matrix = X.corr(numeric_only=True).round(2)
    c1, c2 = st.columns([1.5, 1])
    with c1:
        fig = plt.figure(figsize=(10, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
        plt.title("Matrice de corrélation")
        montrer(fig)
    with c2:
        cm = correlation_matrix
        st.markdown(f"""
Plusieurs variables disent à peu près la même chose :

- free_sulfur_dioxide et total_sulfur_dioxide : **{fr(cm.loc['free_sulfur_dioxide', 'total_sulfur_dioxide'], 2)}**
- density et alcohol : **{fr(cm.loc['density', 'alcohol'], 2)}**
- density et residual_sugar : **{fr(cm.loc['density', 'residual_sugar'], 2)}**
- density et fixed_acidity : **{fr(cm.loc['density', 'fixed_acidity'], 2)}**

Logique : l'alcool est moins dense que l'eau, le sucre plus dense.

Cette multicolinéarité peut gêner la régression logistique et Naive Bayes.
On teste donc deux pistes : l'ACP et la sélection de variables.
""")

# ---------------------------------------------------------------------------
elif page == "ACP":
    st.title("Analyse en composantes principales")
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA().fit(X_scaled)
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    n_components_90 = int(np.argmax(cumulative_variance >= 0.90)) + 1

    fig = plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(explained_variance_ratio) + 1), cumulative_variance, marker="o", linestyle="--")
    plt.axhline(y=0.90, color="r", linestyle=":", label="Seuil 90%")
    plt.xlabel("Nombre de composantes principales")
    plt.ylabel("Variance expliquée cumulée")
    plt.title("Graphique de la variance expliquée par l'ACP")
    plt.legend()
    plt.grid(True)
    c1, c2 = st.columns([1.4, 1])
    with c1:
        montrer(fig)
    with c2:
        st.write(f"Il faut **{n_components_90} composantes sur 12** pour garder 90 % de la variance. "
                 "La réduction est faible : les variables ne sont pas assez redondantes pour être résumées en peu d'axes.")
        st.caption("ACP faite ici sur tout le jeu, uniquement pour explorer. Dans les modèles, elle est apprise "
                   "dans le pipeline, sur le train.")
# ---------------------------------------------------------------------------
elif page == "Sélection de variables":
    st.title("Sélection de variables")
    st.write("Trois critères, calculés **sur le train uniquement** : la corrélation avec la cible, les "
             "coefficients d'une régression logistique (variables standardisées) et l'importance dans un Random Forest.")
    selection = criteres_selection()
    c1, c2 = st.columns([1, 1.3])
    with c1:
        st.dataframe(selection.round(3).sort_values("Importance_RF", ascending=False), height=460)
    with c2:
        axes = selection.plot(kind="bar", subplots=True, figsize=(10, 9), legend=False)
        fig = axes[0].get_figure()
        plt.tight_layout()
        montrer(fig)
    st.markdown("""
Les trois critères s'accordent sur le haut du classement : **alcohol** est la variable la plus importante, suivie de **density**,
puis chlorides et volatile_acidity, qui jouent négativement. is_red n'apporte quasiment rien.
Attention aux coefficients de la régression logistique : residual_sugar et fixed_acidity y ont un poids fort alors qu'ils sont
peu corrélés à la cible. C'est la multicolinéarité avec density qui joue, pas un vrai effet.
""")

    st.subheader("Est-ce que réduire aide vraiment ?")
    comp, top8 = comparer_jeux_variables()
    st.table(comp)
    st.caption("F1 moyen ± écart-type en validation croisée 5 plis, hyperparamètres par défaut. "
               "Variables gardées par le Random Forest : " + ", ".join(top8) + ".")
    st.markdown("""
- L'**ACP fait moins bien pour les cinq modèles** : elle mélange les variables et perd de l'information utile pour la classe.
- La **sélection** donne à peu près la même chose que tout garder, l'écart reste dans le bruit.

On garde donc **les 12 variables**.
""")

# ---------------------------------------------------------------------------
elif page == "Les cinq modèles":
    st.title("Les cinq modèles")
    st.table(pd.DataFrame({
        "Famille": ["Linéaire", "Non linéaire, à distance", "Non linéaire, à règles", "Probabiliste", "Non linéaire, à noyau"],
        "Ce qui joue pour lui": ["Simple, interprétable : notre référence",
                                 "Aucune hypothèse sur la forme des données",
                                 "Pas besoin de mise à l'échelle, insensible aux extrêmes",
                                 "Très rapide",
                                 "Frontière courbe, C règle la tolérance au bruit"],
        "Ce qui joue contre lui ici": ["Frontière droite, gêné par la multicolinéarité",
                                       "Sensible à l'échelle, ne peut pas pondérer les classes",
                                       "Sur-apprend vite si on ne limite pas sa profondeur",
                                       "Suppose des variables indépendantes et gaussiennes : faux ici",
                                       "Sensible à l'échelle, boîte noire"]},
        index=MODELES))
    st.markdown("""
**Règles communes**

- Chaque modèle est dans un `Pipeline` avec le RobustScaler, pour que la mise à l'échelle soit réapprise à chaque pli de validation croisée.
- `class_weight='balanced'` dès que le modèle l'accepte (tous sauf k-NN et Naive Bayes) : une erreur sur un bon vin coûte environ 4 fois plus.
- On juge sur le **F1 de la classe « bon vin »** et sur l'**AUC**. L'accuracy ne veut rien dire ici : répondre « mauvais » à chaque fois donne déjà 81 %.
""")

# ---------------------------------------------------------------------------
elif page == "Résultats":
    st.title("Résultats sur le jeu de test")
    version = st.radio("Hyperparamètres", ["Par défaut", "Optimisés"], horizontal=True)
    opt = version == "Optimisés"
    titre = "modèles optimisés" if opt else "modèles par défaut"
    res = tableau_test(opt)
    mods = entrainer(opt)
    if opt:
        with st.expander("Comment on les a optimisés"):
            st.write("Recherche sur le train en validation croisée 5 plis, en maximisant le F1. "
                     "Grid Search quand la grille est petite, Random Search pour le SVM (36 combinaisons, on en tire 15).")
            st.table(GRILLES)

    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.dataframe(res.round(3))
        best_f1, best_auc, best_acc = res["F1"].idxmax(), res["AUC"].idxmax(), res["Accuracy"].idxmax()
        st.markdown(f"""
- Meilleur F1 : **{best_f1}** ({fr(res.loc[best_f1, 'F1'])}). Meilleure AUC : **{best_auc}** ({fr(res.loc[best_auc, 'AUC'])}).
- Meilleure accuracy : {best_acc} ({fr(res.loc[best_acc, 'Accuracy'])}), ce qui ne veut pas dire grand-chose ici.
  Le k-NN a un rappel de {fr(res.loc['k-NN', 'Rappel'])} : sans `class_weight`, il penche vers la classe majoritaire et rate des bons vins.
- SVM et régression logistique font l'inverse : rappel autour de {fr(res.loc['SVM', 'Rappel'], 2)}, précision autour de
  {fr(res.loc['SVM', 'Précision'], 2)}. Ils trouvent presque tous les bons vins mais avec beaucoup de fausses alertes.
""")
    with c2:
        montrer(fig_roc(mods, titre))
    montrer(fig_confusions(mods, titre))

    if opt:
        base = tableau_test(False)
        st.subheader("Avant / après optimisation")
        ecart = pd.DataFrame({"F1 avant": base["F1"], "F1 après": res["F1"],
                              "AUC avant": base["AUC"], "AUC après": res["AUC"]}).round(3)
        st.dataframe(ecart)
        g = (res["AUC"] - base["AUC"]).idxmax()
        st.write(f"Peu de choses, sauf pour ce modèle : **{g}** (AUC {fr(base.loc[g, 'AUC'])} → {fr(res.loc[g, 'AUC'])}) "
                 "qu'on a fortement bridé. Les autres étaient déjà proches de leur optimum : la limite vient des données, "
                 "pas du réglage.")

# ---------------------------------------------------------------------------
elif page == "Biais / variance":
    st.title("Compromis biais / variance")
    bv = biais_variance()
    c1, c2 = st.columns([1.3, 1])
    with c1:
        ax = bv[["F1 train", "F1 validation croisée", "F1 test"]].plot(kind="bar", figsize=(10, 5), rot=15)
        plt.ylabel("F1-score")
        plt.title("F1 sur le train, en validation croisée et sur le test")
        plt.grid(axis="y")
        montrer(ax.get_figure())
    with c2:
        st.dataframe(bv.round(3))
    st.markdown(f"""
- **Régression logistique, Naive Bayes** : même score partout, autour de 0,5. Le modèle est trop simple pour les données → **biais**.
- **k-NN** : F1 de {fr(bv.loc['k-NN', 'F1 train'], 2)} sur le train. Avec des poids par distance, chaque vin du train est son
  propre voisin, donc ce chiffre est mécanique. Mais l'écart avec la CV ({fr(bv.loc['k-NN', 'F1 validation croisée'])}) montre bien qu'il colle au train → **variance**.
- **Arbre** : on l'a tellement bridé (profondeur 3) qu'il est passé du côté du biais.
- **SVM** : le meilleur score en CV, avec un écart train / CV de seulement {fr(bv.loc['SVM', 'Écart train - CV'])}. C'est le meilleur compromis.
""")

# ---------------------------------------------------------------------------
elif page == "Conclusion":
    st.title("Ce qu'on retient")
    res = tableau_test(True)
    s = res.loc["SVM"]
    st.markdown(f"""
### Modèle retenu : SVM 
RobustScaler, `class_weight='balanced'`, C = 0,5, gamma = 0,2.

On l'a choisi sur la **validation croisée** (meilleur F1 et meilleur compromis biais / variance), et le test le confirme :
AUC **{fr(s['AUC'])}**, F1 **{fr(s['F1'])}**, il retrouve **{fr(100 * s['Rappel'], 0)} %** des bons vins
avec une précision de **{fr(100 * s['Précision'], 0)} %**.

### Pourquoi on ne fait pas mieux
- La note vient d'une **dégustation** : une partie ne s'explique pas par la chimie.
- **{fr(100 * D['y_train'].mean(), 0)} %** de bons vins seulement.
- Hormis l'alcool, les liens entre les mesures et la note sont faibles.

### Ce qu'on a appris en route
- Les **doublons** faussaient nos premiers résultats.
- L'**accuracy** était trompeuse, le F1 et l'AUC sont plus honnêtes.
- L'**ACP** et la sélection de variables n'ont rien apporté ici, et c'était mesurable.


""")

else:
    st.title("Prédire un vin")
    st.caption("Modèle utilisé : le SVM optimisé (C = 0,5, gamma = 0,2).")
    svm = entrainer(True)["SVM"]
    Xtr, Xte, yte = D["X_train"], D["X_test"], D["y_test"]
    bornes = {f: (float(X[f].min()), float(X[f].max()), float(Xtr[f].median())) for f in MESURES}

    for f in MESURES:
        st.session_state.setdefault(f"s_{f}", bornes[f][2])
    st.session_state.setdefault("s_is_red", "Blanc")
    st.session_state.setdefault("vrai", None)

    def tirer(classe):
        idx = yte[yte == classe].sample(1).index[0]
        ligne = Xte.loc[idx]
        for f in MESURES:
            st.session_state[f"s_{f}"] = float(ligne[f])
        st.session_state["s_is_red"] = "Rouge" if ligne["is_red"] == 1 else "Blanc"
        st.session_state["vrai"] = (classe, int(D["quality"].loc[idx]))

    def medianes():
        for f in MESURES:
            st.session_state[f"s_{f}"] = bornes[f][2]
        st.session_state["s_is_red"] = "Blanc"
        st.session_state["vrai"] = None

    b1, b2, b3 = st.columns(3)
    b1.button("🍷 Tirer un bon vin", on_click=tirer, args=(1,), use_container_width=True)
    b2.button("Tirer un mauvais vin", on_click=tirer, args=(0,), use_container_width=True)
    b3.button("↺ Remettre les médianes", on_click=medianes, use_container_width=True)
    st.divider()

    GROUPES = {
        "Acidité": ["fixed_acidity", "volatile_acidity", "citric_acid", "pH"],
        "Sucre, sel, soufre": ["residual_sugar", "chlorides", "free_sulfur_dioxide",
                               "total_sulfur_dioxide", "sulphates"],
        "Alcool et densité": ["alcohol", "density"],
    }

    col_in, col_out = st.columns([1.3, 1], gap="large")
    with col_in:
        st.radio("Type de vin", ["Blanc", "Rouge"], key="s_is_red", horizontal=True)
        valeurs = {}
        onglets = st.tabs(list(GROUPES))
        for onglet, vars_groupe in zip(onglets, GROUPES.values()):
            with onglet:
                for f in vars_groupe:
                    lo, hi, _ = bornes[f]
                    pas = float(10 ** np.floor(np.log10((hi - lo) / 200)))
                    valeurs[f] = st.slider(NOMS[f], lo, hi, key=f"s_{f}", step=pas, format="%.4g")
        valeurs["is_red"] = 1 if st.session_state["s_is_red"] == "Rouge" else 0

    entree = pd.DataFrame([valeurs])[FEATURES]
    pred = int(svm.predict(entree)[0])
    dec = float(svm.decision_function(entree)[0])

    with col_out:
        couleur = "seagreen" if pred == 1 else "salmon"
        texte = "Bon vin" if pred == 1 else "Vin ordinaire"
        sur = "sûr de lui" if abs(dec) > 1 else "hésitant"
        st.markdown(f"""
<div style="background:{couleur};color:white;padding:22px;border-radius:12px;text-align:center;">
  <div style="font-size:30px;font-weight:700;">{texte}</div>
  <div style="font-size:15px;margin-top:6px;">Score du SVM : {dec:+.2f} · modèle {sur}</div>
</div>""", unsafe_allow_html=True)
        st.caption("Score positif = bon vin. Plus il est loin de 0, plus le modèle est sûr.")

        vrai = st.session_state["vrai"]
        if vrai is not None:
            classe, note = vrai
            if classe == pred:
                st.success(f"Vraie note : {note}/10. Le modèle a raison.")
            else:
                st.error(f"Vraie note : {note}/10. Le modèle se trompe.")

        st.markdown("**Ce vin face aux médianes du train**")
        med_b = Xtr[D["y_train"] == 1].median()
        med_m = Xtr[D["y_train"] == 0].median()
        cles = ["alcohol", "density", "volatile_acidity", "chlorides", "total_sulfur_dioxide"]
        comp = pd.DataFrame({"Ce vin": [valeurs[c] for c in cles],
                             "Bons vins": [med_b[c] for c in cles],
                             "Autres": [med_m[c] for c in cles]},
                            index=[NOMS[c] for c in cles])

        def colorer(ligne):
            proche_bon = abs(ligne["Ce vin"] - ligne["Bons vins"]) < abs(ligne["Ce vin"] - ligne["Autres"])
            c = "background-color: rgba(46,139,87,0.25)" if proche_bon else "background-color: rgba(250,128,114,0.25)"
            return [c, "", ""]

        st.dataframe(comp.style.apply(colorer, axis=1).format("{:.4g}"), use_container_width=True)
        st.caption("Vert : valeur plus proche des bons vins. Rouge : plus proche des autres.")