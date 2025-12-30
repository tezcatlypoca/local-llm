"""
Script de lancement de l'API de base (inférence LLM).
Version sans rate limiting pour usage interne.
"""

import sys
from pathlib import Path

# Ajouter le répertoire racine au PYTHONPATH
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Lancer l'application de base (sans rate limiting)
if __name__ == '__main__':
    from src.main_base import app
    app.run(host='0.0.0.0', port=5000, debug=True)

