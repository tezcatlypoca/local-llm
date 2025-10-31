#!/bin/bash
"""
Script pour créer et activer un service systemd pour l'API Local LLM.

Usage:
    sudo bash scripts/setup_systemd_service.sh
"""

set -e

# Couleurs pour les messages
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Configuration du service systemd pour l'API Local LLM"
echo "=========================================="
echo

# Récupérer le chemin du projet
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Récupérer l'utilisateur actuel
CURRENT_USER=$(whoami)

echo "📁 Dossier du projet: $PROJECT_DIR"
echo "👤 Utilisateur: $CURRENT_USER"
echo

# Vérifier qu'on est root ou sudo
if [ "$EUID" -ne 0 ]; then 
    echo -e "${YELLOW}⚠️  Attention: Ce script doit être exécuté avec sudo${NC}"
    echo "   Utilisez: sudo bash scripts/setup_systemd_service.sh"
    exit 1
fi

# Demander confirmation
read -p "Voulez-vous créer le service systemd ? (o/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[OoYy]$ ]]; then
    echo "❌ Annulé"
    exit 1
fi

# Trouver le chemin Python du venv
VENV_PYTHON="$PROJECT_DIR/venv/bin/python"
if [ ! -f "$VENV_PYTHON" ]; then
    echo -e "${YELLOW}⚠️  Avertissement: venv/bin/python non trouvé${NC}"
    echo "   Le service utilisera 'python3' du système"
    PYTHON_CMD="python3"
else
    PYTHON_CMD="$VENV_PYTHON"
    echo "✅ Python du venv trouvé: $PYTHON_CMD"
fi

# Créer le fichier de service
SERVICE_FILE="/etc/systemd/system/local-llm-api.service"

echo
echo "📝 Création du fichier de service..."

cat > "$SERVICE_FILE" << EOF
[Unit]
Description=Local LLM API Service
After=network.target

[Service]
Type=simple
User=$CURRENT_USER
WorkingDirectory=$PROJECT_DIR
Environment="PATH=$PROJECT_DIR/venv/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=$PYTHON_CMD $PROJECT_DIR/src/main.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

echo "✅ Fichier de service créé: $SERVICE_FILE"
echo

# Recharger systemd
echo "🔄 Rechargement de systemd..."
systemctl daemon-reload
echo "✅ systemd rechargé"
echo

# Demander si on veut activer et démarrer le service
read -p "Voulez-vous activer le service (démarrage automatique) ? (o/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[OoYy]$ ]]; then
    systemctl enable local-llm-api.service
    echo "✅ Service activé (démarrage automatique)"
    
    read -p "Voulez-vous démarrer le service maintenant ? (o/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[OoYy]$ ]]; then
        systemctl start local-llm-api.service
        echo "✅ Service démarré"
        echo
        echo "📊 Statut du service:"
        systemctl status local-llm-api.service --no-pager
    fi
fi

echo
echo -e "${GREEN}✅ Configuration terminée !${NC}"
echo
echo "📋 Commandes utiles:"
echo "   sudo systemctl status local-llm-api    # Voir le statut"
echo "   sudo systemctl start local-llm-api     # Démarrer le service"
echo "   sudo systemctl stop local-llm-api      # Arrêter le service"
echo "   sudo systemctl restart local-llm-api   # Redémarrer le service"
echo "   sudo systemctl disable local-llm-api   # Désactiver le démarrage auto"
echo "   sudo journalctl -u local-llm-api -f    # Voir les logs en temps réel"
echo

