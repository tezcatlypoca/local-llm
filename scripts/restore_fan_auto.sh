#!/bin/bash
"""
Script pour restaurer le contrôle automatique des ventilateurs GPU AMD.

Ce script restaure le mode automatique des ventilateurs pour tous les GPUs AMD.

Usage:
    sudo bash scripts/restore_fan_auto.sh
"""

set -e

# Couleurs pour les messages
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Vérifier qu'on est root
if [ "$EUID" -ne 0 ]; then 
    echo -e "${YELLOW}⚠️  Ce script doit être exécuté avec sudo${NC}"
    echo "   Utilisez: sudo bash scripts/restore_fan_auto.sh"
    exit 1
fi

echo "=========================================="
echo "Restauration du contrôle automatique"
echo "=========================================="
echo

# Fonction pour trouver les chemins hwmon des GPUs AMD
find_gpu_hwmon_paths() {
    local paths=()
    for card in /sys/class/drm/card*/device/hwmon/hwmon*/; do
        if [ -d "$card" ] && [ -f "$card/pwm1_enable" ]; then
            if [ -f "$card/name" ]; then
                local name=$(cat "$card/name" 2>/dev/null || echo "")
                if [[ "$name" == *"amdgpu"* ]] || [[ "$name" == *"radeon"* ]] || [ -f "$card/temp1_input" ]; then
                    paths+=("$card")
                fi
            elif [ -f "$card/temp1_input" ]; then
                paths+=("$card")
            fi
        fi
    done
    echo "${paths[@]}"
}

# Restaurer le mode automatique
RESTORED_COUNT=0
HWMON_PATHS=($(find_gpu_hwmon_paths))

if [ ${#HWMON_PATHS[@]} -eq 0 ]; then
    echo -e "${RED}❌ Aucun GPU AMD détecté${NC}"
    exit 1
fi

for hwmon_path in "${HWMON_PATHS[@]}"; do
    # Chercher tous les fichiers pwm*_enable
    for i in {1..5}; do
        pwm_enable_file="$hwmon_path/pwm${i}_enable"
        if [ -f "$pwm_enable_file" ]; then
            # Activer le mode automatique (2 = automatique)
            if echo "2" > "$pwm_enable_file" 2>/dev/null; then
                echo -e "${GREEN}✅ Mode automatique restauré pour $hwmon_path (pwm${i})${NC}"
                ((RESTORED_COUNT++))
            fi
        fi
    done
done

if [ $RESTORED_COUNT -eq 0 ]; then
    echo -e "${YELLOW}⚠️  Aucun ventilateur n'a pu être restauré${NC}"
    exit 1
fi

echo
echo "=========================================="
echo -e "${GREEN}✅ Restauration terminée${NC}"
echo "=========================================="
echo "   - ${RESTORED_COUNT} ventilateur(s) restauré(s) en mode automatique"
echo

