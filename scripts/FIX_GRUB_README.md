# Guide pour configurer GRUB pour booter sur kernel 6.11.0.21

## Solution automatique (recommandée)

Exécutez le script fourni :

```bash
sudo bash scripts/fix_grub_kernel_boot.sh
```

## Solution manuelle rapide

Si vous préférez le faire manuellement, voici les étapes :

### 1. Éditer la configuration GRUB

```bash
sudo nano /etc/default/grub
```

### 2. Modifier la ligne GRUB_DEFAULT

Trouvez la ligne `GRUB_DEFAULT=0` et remplacez-la par :

```bash
GRUB_DEFAULT="Advanced options for Ubuntu>Ubuntu, with Linux 6.11.0-21-generic"
```

Ou si vous préférez utiliser l'index numérique :

```bash
GRUB_DEFAULT="1>2"
```

(Pour trouver l'index exact, utilisez `grep "menuentry" /boot/grub/grub.cfg | grep -n "6.11.0-21"`)

### 3. Mettre à jour GRUB

```bash
sudo update-grub
```

### 4. Redémarrer

```bash
sudo reboot
```

## Solution alternative : Utiliser grub-set-default

Si vous avez déjà booté sur le kernel 6.11.0.21 :

```bash
# Définir le kernel actuel comme défaut
sudo grub-set-default "Advanced options for Ubuntu>Ubuntu, with Linux 6.11.0-21-generic"

# Ou utiliser l'index du menu
sudo grub-set-default "1>2"  # (remplacez 1>2 par l'index correct)

# Mettre à jour GRUB
sudo update-grub
```

## Supprimer le kernel problématique (optionnel)

Pour éviter que le kernel 6.14.0.33 ne réapparaisse dans le menu :

```bash
# Voir les kernels installés
dpkg -l | grep linux-image

# Supprimer le kernel problématique
sudo apt autoremove --purge linux-image-6.14.0-33-generic linux-headers-6.14.0-33-generic

# Mettre à jour GRUB
sudo update-grub
```

## Vérifier la configuration

Pour vérifier quel kernel est configuré par défaut :

```bash
# Voir la configuration GRUB actuelle
grep GRUB_DEFAULT /etc/default/grub

# Voir le kernel par défaut actuel
grub-editenv list | grep saved_entry
```

## Restaurer la configuration précédente

Si quelque chose ne va pas, vous pouvez restaurer le backup :

```bash
# Trouver le backup
ls -la /etc/default/grub.backup.*

# Restaurer (remplacez par le nom du fichier backup)
sudo cp /etc/default/grub.backup.YYYYMMDD_HHMMSS /etc/default/grub
sudo update-grub
```

## Notes importantes

- Le script crée automatiquement un backup de votre configuration GRUB
- Vous devrez redémarrer pour que les changements prennent effet
- Si le kernel 6.11.0.21 n'apparaît pas dans le menu, vérifiez qu'il est toujours installé : `dpkg -l | grep linux-image-6.11.0-21`

