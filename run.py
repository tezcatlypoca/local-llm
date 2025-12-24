"""
Script de lancement de l'application Flask.
Ce script configure le PYTHONPATH et lance l'application.
"""

import sys
from pathlib import Path

# Ajouter le répertoire racine au PYTHONPATH
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Lancer l'application
if __name__ == '__main__':
    from src.main import app
    app.run(host='0.0.0.0', port=5000, debug=True)

