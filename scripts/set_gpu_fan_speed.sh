#!/bin/bash
"""
Script pour configurer la vitesse minimale des ventilateurs des GPUs AMD.

Ce script configure les ventilateurs des GPUs AMD Vega 64 à une vitesse minimale
de 20-30% pour éviter les pics de température lors des charges d'inférence.

Usage:
    sudo bash scripts/set_gpu_fan_speed.sh [speed_percent]
    
    speed_percent: Pourcentage de vitesse du ventilateur (20-100, défaut: 25)
"""

# Ne pas utiliser set -e pour permettre la gestion d'erreurs manuelle
# set -e

# Couleurs pour les messages
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Vitesse par défaut (25%)
# Si le premier argument commence par "--", c'est une option
if [[ "${1:-}" == --wait ]]; then
    WAIT_FOR_GPUS=true
    FAN_SPEED_PERCENT=${2:-25}
elif [[ "${1:-}" == --* ]]; then
    echo -e "${RED}❌ Option inconnue: $1${NC}"
    echo "Usage: $0 [--wait] [speed_percent]"
    exit 1
else
    WAIT_FOR_GPUS=false
    FAN_SPEED_PERCENT=${1:-25}
fi

# Si INVOCATION_ID est définie (systemd), activer l'attente automatiquement
if [ -n "$INVOCATION_ID" ]; then
    WAIT_FOR_GPUS=true
fi

# Validation de la vitesse
if ! [[ "$FAN_SPEED_PERCENT" =~ ^[0-9]+$ ]] || [ "$FAN_SPEED_PERCENT" -lt 20 ] || [ "$FAN_SPEED_PERCENT" -gt 100 ]; then
    echo -e "${RED}❌ Erreur: La vitesse doit être un nombre entre 20 et 100${NC}"
    exit 1
fi

# Vérifier qu'on est root
if [ "$EUID" -ne 0 ]; then 
    echo -e "${YELLOW}⚠️  Ce script doit être exécuté avec sudo${NC}"
    echo "   Utilisez: sudo bash scripts/set_gpu_fan_speed.sh $FAN_SPEED_PERCENT"
    exit 1
fi

echo "=========================================="
echo "Configuration des ventilateurs GPU AMD"
echo "=========================================="
echo "Vitesse cible: ${FAN_SPEED_PERCENT}%"
echo

# Fonction pour trouver les chemins hwmon des GPUs AMD
find_gpu_hwmon_paths() {
    local paths=()
    for card in /sys/class/drm/card*/device/hwmon/hwmon*/; do
        if [ -d "$card" ] && [ -f "$card/pwm1_enable" ]; then
            # Vérifier que c'est bien un GPU AMD (présence de name ou temp1_input)
            if [ -f "$card/name" ]; then
                local name=$(cat "$card/name" 2>/dev/null || echo "")
                if [[ "$name" == *"amdgpu"* ]] || [[ "$name" == *"radeon"* ]] || [ -f "$card/temp1_input" ]; then
                    paths+=("$card")
                fi
            elif [ -f "$card/temp1_input" ]; then
                # Si pas de name mais temp1_input présent, probablement un GPU
                paths+=("$card")
            fi
        fi
    done
    echo "${paths[@]}"
}

# Fonction pour configurer un ventilateur
set_fan_speed() {
    local hwmon_path=$1
    local speed_percent=$2
    
    # Convertir le pourcentage en valeur PWM (0-255) - calcul bash natif
    local pwm_value=$((speed_percent * 255 / 100))
    
    # Trouver le fichier PWM approprié (pwm1, pwm2, etc.)
    local pwm_file=""
    local pwm_enable_file=""
    
    # Chercher pwm1, pwm2, etc. jusqu'à pwm5
    for i in {1..5}; do
        if [ -f "$hwmon_path/pwm${i}_enable" ] && [ -f "$hwmon_path/pwm${i}" ]; then
            pwm_file="$hwmon_path/pwm${i}"
            pwm_enable_file="$hwmon_path/pwm${i}_enable"
            
            # Vérifier si c'est le ventilateur du GPU (pas d'autres capteurs)
            # Les ventilateurs GPU sont généralement sur pwm1 ou pwm2
            local current_mode=$(cat "$pwm_enable_file" 2>/dev/null || echo "0")
            
            echo -e "${YELLOW}📌 Configuration de $hwmon_path (pwm${i})${NC}"
            
            # Sauvegarder le mode actuel (si on veut le restaurer plus tard)
            echo "$current_mode" > "$hwmon_path/pwm${i}_enable.backup" 2>/dev/null || true
            
            # Activer le contrôle manuel (1 = manuel, 2 = automatique)
            echo "1" > "$pwm_enable_file" 2>/dev/null || {
                echo -e "${RED}❌ Impossible d'activer le contrôle manuel pour $hwmon_path${NC}"
                continue
            }
            
            # Définir la vitesse
            echo "$pwm_value" > "$pwm_file" 2>/dev/null || {
                echo -e "${RED}❌ Impossible de définir la vitesse pour $hwmon_path${NC}"
                continue
            }
            
            # Lire la valeur pour vérification
            local actual_value=$(cat "$pwm_file" 2>/dev/null || echo "0")
            # Calculer le pourcentage (approximation simple)
            local actual_percent=$((actual_value * 100 / 255))
            
            echo -e "${GREEN}✅ Ventilateur configuré: ~${actual_percent}% (PWM: $actual_value/255)${NC}"
            
            # Afficher la température actuelle si disponible
            if [ -f "$hwmon_path/temp1_input" ]; then
                local temp=$(cat "$hwmon_path/temp1_input" 2>/dev/null || echo "0")
                temp=$((temp / 1000))  # Convertir de millidegrés à degrés
                echo "   Température actuelle: ${temp}°C"
            fi
            
            return 0
        fi
    done
    
    echo -e "${RED}❌ Aucun fichier PWM trouvé dans $hwmon_path${NC}"
    return 1
}

# Fonction pour attendre que les GPUs soient prêts (avec retry)
wait_for_gpus() {
    local max_attempts=30
    local attempt=0
    local wait_seconds=2
    
    while [ $attempt -lt $max_attempts ]; do
        HWMON_PATHS=($(find_gpu_hwmon_paths))
        if [ ${#HWMON_PATHS[@]} -gt 0 ]; then
            return 0
        fi
        
        attempt=$((attempt + 1))
        if [ $attempt -lt $max_attempts ]; then
            echo -e "${YELLOW}⏳ Attente des GPUs... (tentative $attempt/$max_attempts)${NC}"
            sleep $wait_seconds
        fi
    done
    
    return 1
}

# Trouver tous les GPUs AMD (avec retry si nécessaire)
if [ "$WAIT_FOR_GPUS" = "true" ]; then
    if ! wait_for_gpus; then
        echo -e "${YELLOW}⚠️  Aucun GPU AMD détecté après attente${NC}"
        echo "Le service continuera pour permettre un redémarrage automatique"
        # Ne pas faire exit 1 ici pour le service, laisser systemd gérer
        HWMON_PATHS=()
    fi
else
    HWMON_PATHS=($(find_gpu_hwmon_paths))
fi

if [ ${#HWMON_PATHS[@]} -eq 0 ]; then
    echo -e "${RED}❌ Aucun GPU AMD détecté (pas de fichiers hwmon trouvés)${NC}"
    echo
    echo "Vérifications possibles:"
    echo "1. Vérifiez que les drivers amdgpu sont chargés: lsmod | grep amdgpu"
    echo "2. Vérifiez les chemins: ls -la /sys/class/drm/card*/device/hwmon/"
    echo "3. Vérifiez les permissions: vous devez être root"
    echo
    # Si pas de GPU, exit avec code 0 pour ne pas faire échouer le service
    # (peut être utile si les GPUs ne sont pas encore initialisés)
    echo -e "${YELLOW}⚠️  Sortie avec code 0 (service peut être relancé)${NC}"
    exit 0
fi

echo -e "${GREEN}✅ ${#HWMON_PATHS[@]} GPU(s) AMD détecté(s)${NC}"
echo

# Configurer chaque GPU
SUCCESS_COUNT=0
for hwmon_path in "${HWMON_PATHS[@]}"; do
    if set_fan_speed "$hwmon_path" "$FAN_SPEED_PERCENT"; then
        ((SUCCESS_COUNT++))
    fi
    echo
done

if [ $SUCCESS_COUNT -eq 0 ]; then
    echo -e "${YELLOW}⚠️  Aucun ventilateur n'a pu être configuré${NC}"
    echo "Cela peut être normal si les GPUs ne sont pas encore prêts"
    # Exit avec code 0 pour permettre au service de se relancer automatiquement
    exit 0
fi

echo "=========================================="
echo -e "${GREEN}✅ Configuration terminée${NC}"
echo "=========================================="
echo
echo "📋 Informations:"
echo "   - ${SUCCESS_COUNT} ventilateur(s) configuré(s) à ${FAN_SPEED_PERCENT}%"
echo "   - Le contrôle est maintenant en mode MANUEL"
echo
echo "⚠️  IMPORTANT:"
echo "   - Les ventilateurs resteront à cette vitesse même au repos"
echo "   - Surveillez les températures avec: watch -n 1 rocm-smi"
echo "   - Pour revenir au mode automatique, utilisez le script restore_fan_auto.sh"
echo
echo "📊 Vérifier la configuration:"
echo "   rocm-smi -a"
echo "   # ou"
echo "   watch -n 1 'cat /sys/class/drm/card*/device/hwmon/hwmon*/pwm1'"
echo

