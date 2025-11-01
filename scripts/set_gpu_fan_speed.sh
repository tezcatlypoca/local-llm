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
    # Les ventilateurs GPU sont généralement sur pwm1 ou pwm2
    
    # Chercher d'abord pwm1 (le plus commun pour les ventilateurs GPU)
    local found_pwm=false
    
    for i in {1..5}; do
        local pwm_file="$hwmon_path/pwm${i}"
        local pwm_enable_file="$hwmon_path/pwm${i}_enable"
        
        if [ -f "$pwm_enable_file" ] && [ -f "$pwm_file" ]; then
            # Vérifier s'il y a un fichier fan correspondant (fan1_input, etc.)
            local has_fan_input=false
            if [ -f "$hwmon_path/fan${i}_input" ]; then
                has_fan_input=true
            fi
            
            # Si c'est pwm1 ou si c'est un PWM avec fan_input, c'est probablement le ventilateur
            if [ "$i" = "1" ] || [ "$has_fan_input" = "true" ]; then
                found_pwm=true
                
                echo -e "${YELLOW}📌 Configuration de $hwmon_path (pwm${i})${NC}"
                
                # Vérifier les limites PWM si disponibles
                local pwm_min=0
                local pwm_max=255
                if [ -f "$hwmon_path/pwm${i}_min" ]; then
                    pwm_min=$(cat "$hwmon_path/pwm${i}_min" 2>/dev/null || echo "0")
                    echo "   PWM min disponible: $pwm_min"
                fi
                if [ -f "$hwmon_path/pwm${i}_max" ]; then
                    pwm_max=$(cat "$hwmon_path/pwm${i}_max" 2>/dev/null || echo "255")
                    echo "   PWM max disponible: $pwm_max"
                fi
                
                # S'assurer que la valeur est dans les limites
                if [ "$pwm_value" -lt "$pwm_min" ]; then
                    echo -e "${YELLOW}⚠️  Valeur PWM $pwm_value < min $pwm_min, utilisation de $pwm_min${NC}"
                    pwm_value=$pwm_min
                fi
                if [ "$pwm_value" -gt "$pwm_max" ]; then
                    echo -e "${YELLOW}⚠️  Valeur PWM $pwm_value > max $pwm_max, utilisation de $pwm_max${NC}"
                    pwm_value=$pwm_max
                fi
                
                # Vérifier le mode actuel
                local current_mode=$(cat "$pwm_enable_file" 2>/dev/null || echo "0")
                echo "   Mode actuel: $current_mode (0=disable, 1=manuel, 2=auto)"
                
                # Sauvegarder le mode actuel
                echo "$current_mode" > "$hwmon_path/pwm${i}_enable.backup" 2>/dev/null || true
                
                # Activer le contrôle manuel (1 = manuel, 2 = automatique)
                # Note: Certains systèmes nécessitent d'écrire "1" deux fois
                if ! echo "1" > "$pwm_enable_file" 2>/dev/null; then
                    echo -e "${RED}❌ Impossible d'activer le contrôle manuel pour pwm${i}${NC}"
                    continue
                fi
                
                # Vérifier que le mode a bien été activé
                sleep 0.1
                local new_mode=$(cat "$pwm_enable_file" 2>/dev/null || echo "0")
                if [ "$new_mode" != "1" ]; then
                    echo -e "${YELLOW}⚠️  Tentative supplémentaire d'activation du mode manuel${NC}"
                    echo "1" > "$pwm_enable_file" 2>/dev/null || true
                    sleep 0.1
                fi
                
                # Définir la vitesse
                if ! echo "$pwm_value" > "$pwm_file" 2>/dev/null; then
                    echo -e "${RED}❌ Impossible de définir la vitesse pour pwm${i}${NC}"
                    continue
                fi
                
                # Lire la valeur pour vérification
                sleep 0.2  # Laisser le temps au matériel de réagir
                local actual_value=$(cat "$pwm_file" 2>/dev/null || echo "0")
                local actual_percent=$((actual_value * 100 / 255))
                
                echo -e "${GREEN}✅ PWM configuré: ~${actual_percent}% (PWM: $actual_value/255)${NC}"
                
                # Vérifier la vitesse réelle du ventilateur si disponible
                if [ -f "$hwmon_path/fan${i}_input" ]; then
                    local fan_speed=$(cat "$hwmon_path/fan${i}_input" 2>/dev/null || echo "0")
                    if [ "$fan_speed" -gt 0 ]; then
                        echo -e "${GREEN}   ✅ Ventilateur en rotation: ${fan_speed} RPM${NC}"
                    else
                        echo -e "${YELLOW}   ⚠️  Ventilateur à 0 RPM - peut nécessiter une valeur PWM plus élevée${NC}"
                        echo "      Essayez d'augmenter la vitesse (ex: 30-40%)"
                    fi
                fi
                
                # Afficher la température actuelle si disponible
                if [ -f "$hwmon_path/temp1_input" ]; then
                    local temp=$(cat "$hwmon_path/temp1_input" 2>/dev/null || echo "0")
                    temp=$((temp / 1000))
                    echo "   Température: ${temp}°C"
                fi
                
                return 0
            fi
        fi
    done
    
    if [ "$found_pwm" = "false" ]; then
        echo -e "${RED}❌ Aucun fichier PWM de ventilateur trouvé dans $hwmon_path${NC}"
        echo "   Fichiers disponibles:"
        ls -la "$hwmon_path"/pwm* 2>/dev/null | head -5 || echo "      (aucun fichier PWM trouvé)"
    fi
    
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
echo "   # Voir les valeurs PWM:"
echo "   cat /sys/class/drm/card*/device/hwmon/hwmon*/pwm1"
echo "   # Voir la vitesse réelle des ventilateurs:"
echo "   cat /sys/class/drm/card*/device/hwmon/hwmon*/fan1_input"
echo "   # Voir les températures:"
echo "   rocm-smi -a"
echo "   # Ou avec watch pour monitoring en temps réel:"
echo "   watch -n 1 'echo \"PWM: \" \$(cat /sys/class/drm/card0/device/hwmon/hwmon*/pwm1 2>/dev/null | head -1) \" RPM: \" \$(cat /sys/class/drm/card0/device/hwmon/hwmon*/fan1_input 2>/dev/null | head -1)'"
echo

