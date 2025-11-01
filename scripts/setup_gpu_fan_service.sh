#!/bin/bash
"""
Script pour créer un service systemd qui configure automatiquement
les ventilateurs GPU au démarrage de la machine.

Usage:
    sudo bash scripts/setup_gpu_fan_service.sh [speed_percent]
    
    speed_percent: Pourcentage de vitesse du ventilateur (20-100, défaut: 25)
"""

set -e

# Couleurs pour les messages
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Vitesse par défaut
FAN_SPEED_PERCENT=${1:-25}

# Validation
if ! [[ "$FAN_SPEED_PERCENT" =~ ^[0-9]+$ ]] || [ "$FAN_SPEED_PERCENT" -lt 20 ] || [ "$FAN_SPEED_PERCENT" -gt 100 ]; then
    echo -e "${RED}❌ Erreur: La vitesse doit être un nombre entre 20 et 100${NC}"
    exit 1
fi

# Vérifier qu'on est root
if [ "$EUID" -ne 0 ]; then 
    echo -e "${YELLOW}⚠️  Ce script doit être exécuté avec sudo${NC}"
    echo "   Utilisez: sudo bash scripts/setup_gpu_fan_service.sh $FAN_SPEED_PERCENT"
    exit 1
fi

echo "=========================================="
echo "Configuration du service GPU Fan"
echo "=========================================="
echo "Vitesse cible: ${FAN_SPEED_PERCENT}%"
echo

# Récupérer le chemin du projet
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Créer le fichier de service
SERVICE_FILE="/etc/systemd/system/gpu-fan-control.service"

echo "📝 Création du fichier de service..."

cat > "$SERVICE_FILE" << EOF
[Unit]
Description=GPU Fan Speed Control (AMD)
After=systemd-udev-settle.service
Wants=systemd-udev-settle.service

[Service]
Type=oneshot
ExecStart=/bin/bash $PROJECT_DIR/scripts/set_gpu_fan_speed.sh --wait $FAN_SPEED_PERCENT
RemainAfterExit=yes
StandardOutput=journal
StandardError=journal
# Redémarrer si le script échoue (peut arriver si GPUs pas encore prêts)
Restart=on-failure
RestartSec=10
# Ne pas redémarrer indéfiniment
StartLimitInterval=300
StartLimitBurst=3

[Install]
WantedBy=multi-user.target
EOF

echo -e "${GREEN}✅ Fichier de service créé: $SERVICE_FILE${NC}"
echo

# Recharger systemd
echo "🔄 Rechargement de systemd..."
systemctl daemon-reload
echo -e "${GREEN}✅ systemd rechargé${NC}"
echo

# Demander si on veut activer le service
read -p "Voulez-vous activer le service (démarrage automatique) ? (o/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[OoYy]$ ]]; then
    systemctl enable gpu-fan-control.service
    echo -e "${GREEN}✅ Service activé (démarrage automatique)${NC}"
    
    read -p "Voulez-vous démarrer le service maintenant ? (o/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[OoYy]$ ]]; then
        systemctl start gpu-fan-control.service
        echo -e "${GREEN}✅ Service démarré${NC}"
        echo
        echo "📊 Statut du service:"
        systemctl status gpu-fan-control.service --no-pager
    fi
fi

echo
echo -e "${GREEN}✅ Configuration terminée !${NC}"
echo
echo "📋 Commandes utiles:"
echo "   sudo systemctl status gpu-fan-control    # Voir le statut"
echo "   sudo systemctl start gpu-fan-control     # Démarrer le service"
echo "   sudo systemctl stop gpu-fan-control      # Arrêter le service"
echo "   sudo systemctl restart gpu-fan-control   # Redémarrer le service"
echo "   sudo systemctl disable gpu-fan-control   # Désactiver le démarrage auto"
echo "   sudo journalctl -u gpu-fan-control -f    # Voir les logs en temps réel"
echo

