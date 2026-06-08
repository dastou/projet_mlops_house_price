# Dockerfile pour l'API Laplace Immo
# Build : docker build -t laplace-immo .
# Run   : docker run -p 8000:8000 laplace-immo
# Puis  : http://localhost:8000/docs

FROM python:3.11-slim

WORKDIR /app

# Installation des dependances Python en premier (layer cache).
# Si seul le code change, pip install n'est pas relance.
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copie du code source et du modele entraine
COPY src/ ./src/
COPY api/ ./api/
COPY models/final_model.pkl ./models/final_model.pkl

# Utilisateur non-root pour la securite (UID 10001 = standard MLOps)
RUN useradd --create-home --uid 10001 app \
    && chown -R app:app /app
USER app

# L'API ecoute sur le port 8000
EXPOSE 8000

# Healthcheck : Docker verifie toutes les 30 s que l'API repond
# python urllib est dispo dans l'image slim, contrairement a curl
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
