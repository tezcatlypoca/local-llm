#!/bin/bash
"""
Script pour configurer GRUB afin de booter toujours sur le kernel 6.11.0.21
au lieu du kernel 6.14.0.33 qui cause un kernel panic.

Usage:
    sudo bash scripts/fix_grub_kernel_boot.sh
"""

set -e

# Couleurs pour les messages
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Configuration GRUB pour kernel 6.11.0.21"
echo "=========================================="
echo

# Vérifier qu'on est root ou sudo
if [ "$EUID" -ne 0 ]; then 
    echo -e "${YELLOW}⚠️  Attention: Ce script doit être exécuté avec sudo${NC}"
    echo "   Utilisez: sudo bash scripts/fix_grub_kernel_boot.sh"
    exit 1
fi

# Vérifier que le kernel 6.11.0.21 existe
KERNEL_611="/boot/vmlinuz-6.11.0-21-generic"
if [ ! -f "$KERNEL_611" ]; then
    echo -e "${YELLOW}⚠️  Avertissement: Kernel 6.11.0.21 non trouvé${NC}"
    echo "   Vérification des kernels disponibles..."
    ls -la /boot/vmlinuz-* 2>/dev/null || echo "   Aucun kernel trouvé dans /boot/"
    echo
    read -p "Voulez-vous continuer quand même ? (o/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[OoYy]$ ]]; then
        echo "❌ Annulé"
        exit 1
    fi
fi

# Sauvegarder la configuration GRUB actuelle
GRUB_CONFIG="/etc/default/grub"
GRUB_BACKUP="/etc/default/grub.backup.$(date +%Y%m%d_%H%M%S)"

if [ -f "$GRUB_CONFIG" ]; then
    cp "$GRUB_CONFIG" "$GRUB_BACKUP"
    echo -e "${GREEN}✅ Configuration GRUB sauvegardée: $GRUB_BACKUP${NC}"
else
    echo -e "${RED}❌ Erreur: Fichier GRUB non trouvé: $GRUB_CONFIG${NC}"
    exit 1
fi

echo
echo "📝 Configuration actuelle de GRUB:"
grep -E "^GRUB_DEFAULT|^GRUB_SAVEDEFAULT" "$GRUB_CONFIG" || echo "   (non définis)"
echo

# Déterminer l'index du kernel 6.11.0.21 dans le menu GRUB
echo "🔍 Recherche de l'index du kernel 6.11.0.21..."
# Générer le menu GRUB temporairement pour trouver l'index
TEMP_MENU=$(mktemp)
grub-mkconfig -o "$TEMP_MENU" 2>/dev/null || grub2-mkconfig -o "$TEMP_MENU" 2>/dev/null || true

if [ -f "$TEMP_MENU" ]; then
    # Chercher la ligne contenant 6.11.0-21
    KERNEL_LINE=$(grep -n "6.11.0-21" "$TEMP_MENU" | head -1)
    if [ -n "$KERNEL_LINE" ]; then
        echo -e "${GREEN}✅ Kernel 6.11.0.21 trouvé dans le menu GRUB${NC}"
        # Extraire le numéro de menuentry (souvent précédé de "menuentry")
        MENU_ENTRY=$(grep -B 5 "6.11.0-21" "$TEMP_MENU" | grep "^menuentry" | head -1 | sed 's/.*'\''\([^'\'']*\)'\''.*/\1/' || echo "")
        if [ -n "$MENU_ENTRY" ]; then
            echo "   Menuentry: $MENU_ENTRY"
        fi
    else
        echo -e "${YELLOW}⚠️  Kernel 6.11.0.21 non trouvé dans le menu GRUB${NC}"
    fi
    rm -f "$TEMP_MENU"
fi

echo
echo "📋 Options de configuration:"
echo "   1. Utiliser le nom du kernel (recommandé)"
echo "   2. Utiliser l'index numérique"
echo "   3. Utiliser 'saved' avec un kernel par défaut"
echo

read -p "Choisissez une option (1-3) [1]: " -n 1 -r
echo

case $REPLY in
    2)
        read -p "Entrez l'index du kernel (commence à 0): " KERNEL_INDEX
        GRUB_DEFAULT_VALUE="'Advanced options for Ubuntu>Ubuntu, with Linux 6.11.0-21-generic'"
        ;;
    3)
        # Utiliser 'saved' et définir le kernel par défaut
        GRUB_DEFAULT_VALUE="saved"
        ;;
    *)
        # Option 1 par défaut: utiliser le nom du kernel
        GRUB_DEFAULT_VALUE="'Advanced options for Ubuntu>Ubuntu, with Linux 6.11.0-21-generic'"
        ;;
esac

# Modifier la configuration GRUB
echo
echo "📝 Modification de la configuration GRUB..."

# Si GRUB_DEFAULT existe déjà, le remplacer, sinon l'ajouter
if grep -q "^GRUB_DEFAULT=" "$GRUB_CONFIG"; then
    sed -i "s/^GRUB_DEFAULT=.*/GRUB_DEFAULT=$GRUB_DEFAULT_VALUE/" "$GRUB_CONFIG"
else
    echo "GRUB_DEFAULT=$GRUB_DEFAULT_VALUE" >> "$GRUB_CONFIG"
fi

# Activer GRUB_SAVEDEFAULT si on utilise 'saved'
if [ "$GRUB_DEFAULT_VALUE" = "saved" ]; then
    if grep -q "^GRUB_SAVEDEFAULT=" "$GRUB_CONFIG"; then
        sed -i "s/^GRUB_SAVEDEFAULT=.*/GRUB_SAVEDEFAULT=true/" "$GRUB_CONFIG"
    else
        echo "GRUB_SAVEDEFAULT=true" >> "$GRUB_CONFIG"
    fi
    
    # Définir le kernel par défaut
    echo "📝 Configuration du kernel par défaut pour 'saved'..."
    grub-set-default "Advanced options for Ubuntu>Ubuntu, with Linux 6.11.0-21-generic" 2>/dev/null || \
    grub2-set-default "Advanced options for Ubuntu>Ubuntu, with Linux 6.11.0-21-generic" 2>/dev/null || \
    echo -e "${YELLOW}⚠️  Impossible de définir le kernel par défaut avec grub-set-default${NC}"
fi

echo -e "${GREEN}✅ Configuration GRUB modifiée${NC}"
echo
echo "📝 Nouvelle configuration:"
grep -E "^GRUB_DEFAULT|^GRUB_SAVEDEFAULT" "$GRUB_CONFIG"
echo

# Mettre à jour GRUB
echo "🔄 Mise à jour de GRUB..."
if command -v update-grub &> /dev/null; then
    update-grub
elif command -v grub-mkconfig &> /dev/null; then
    grub-mkconfig -o /boot/grub/grub.cfg
elif command -v grub2-mkconfig &> /dev/null; then
    grub2-mkconfig -o /boot/grub2/grub.cfg
else
    echo -e "${RED}❌ Erreur: Commande update-grub non trouvée${NC}"
    exit 1
fi

echo
echo -e "${GREEN}✅ Configuration terminée !${NC}"
echo
echo "📋 Résumé des modifications:"
echo "   • Kernel par défaut: 6.11.0-21-generic"
echo "   • Backup de GRUB: $GRUB_BACKUP"
echo
echo "⚠️  IMPORTANT:"
echo "   • Redémarrez votre système pour appliquer les changements"
echo "   • Si le problème persiste, vous pouvez restaurer la config:"
echo "     sudo cp $GRUB_BACKUP $GRUB_CONFIG"
echo "     sudo update-grub"
echo
echo "💡 Pour supprimer le kernel problématique (optionnel):"
echo "   sudo apt autoremove --purge linux-image-6.14.0-33-generic"
echo

