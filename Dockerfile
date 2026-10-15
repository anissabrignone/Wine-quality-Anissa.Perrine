# Image de base Python officielle légère
FROM python:3.11-slim

# Définition du répertoire de travail
WORKDIR /app

# Installation des dépendances système de base
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copie des fichiers de configuration et installation des paquets Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code source et des fichiers du projet
COPY . .

# Exécution préalable de l'entraînement si le modèle n'est pas déjà présent
RUN python train_model.py

# Exposition du port standard de Streamlit
EXPOSE 8501

# Vérification de l'état de santé du conteneur
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Commande de lancement de l'application Streamlit
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
