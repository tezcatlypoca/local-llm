# Réinstallation de PyTorch avec support ROCm

## 🔍 Diagnostic

Si vous obtenez :
- `torch.version.hip = None`
- `torch.cuda.is_available() = False`
- `device_count = 0`

**Cela signifie que PyTorch n'a pas été compilé avec le support ROCm.**

## 📋 Étapes de réinstallation

### 1. Vérifier la version de ROCm installée

```bash
# Méthode 1: Vérifier dans /opt/rocm
ls -la /opt/rocm*/bin/.info/version* 2>/dev/null || cat /opt/rocm/.info/version* 2>/dev/null

# Méthode 2: Vérifier avec rocm-smi (si disponible)
rocm-smi --version

# Méthode 3: Vérifier les bibliothèques
ldconfig -p | grep rocm | head -5
```

**Notez la version de ROCm** (ex: 5.7, 5.6, 6.0, etc.)

---

### 2. Désinstaller PyTorch actuel

```bash
# Activer votre environnement virtuel
source venv/bin/activate  # ou votre environnement

# Désinstaller PyTorch et dépendances
pip uninstall torch torchvision torchaudio -y

# Nettoyer le cache pip (optionnel mais recommandé)
pip cache purge
```

---

### 3. Installer PyTorch avec support ROCm

**Pour ROCm 5.7** (le plus courant) :
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
```

**Pour ROCm 5.6** :
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6
```

**Pour ROCm 6.0** :
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
```

**Si vous ne connaissez pas la version exacte**, essayez ROCm 5.7 en premier (le plus compatible) :
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
```

---

### 4. Vérifier l'installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'HIP: {torch.version.hip}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

**Résultat attendu** :
- `HIP: 5.7.0` (ou une version similaire, pas `None`)
- `CUDA available: True`
- `GPU count: 2` (ou au moins 1)

---

### 5. Si ça ne fonctionne toujours pas

#### A. Vérifier que les bibliothèques ROCm sont accessibles

```bash
# Vérifier que les bibliothèques sont dans le cache
ldconfig -p | grep rocm

# Si vide, ajouter ROCm au cache
sudo ldconfig /opt/rocm/lib
```

#### B. Définir les variables d'environnement

Ajoutez dans votre `~/.bashrc` ou `~/.profile` :

```bash
# Variables ROCm
export ROCM_PATH=/opt/rocm
export HIP_PATH=/opt/rocm
export PATH=$ROCM_PATH/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH

# Pour Vega 64 (gfx900) - IMPORTANT!
export HSA_OVERRIDE_GFX_VERSION=9.0.0
```

Puis rechargez :
```bash
source ~/.bashrc
# ou
source ~/.profile
```

#### C. Vérifier les permissions

```bash
# Vérifier les groupes
groups

# Si vous n'êtes pas dans 'render' et 'video', les ajouter
sudo usermod -a -G render,video $USER

# Redémarrer la session (se déconnecter/reconnecter)
```

#### D. Tester avec le script de diagnostic

```bash
python src/utils/rocm_diagnostic.py
```

---

## 🔧 Installation alternative : Compiler depuis les sources

Si l'installation via pip ne fonctionne pas, vous pouvez essayer d'installer depuis les wheels précompilés ou compiler depuis les sources (plus complexe).

### Option 1: Utiliser conda (si disponible)

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
# Note: Cette commande est pour CUDA, pas ROCm. Pour ROCm avec conda, c'est plus complexe.
```

### Option 2: Vérifier les wheels disponibles

```bash
# Lister les wheels disponibles pour votre version de Python
pip index versions torch --index-url https://download.pytorch.org/whl/rocm5.7
```

---

## 📝 Checklist après installation

- [ ] `torch.version.hip` n'est **pas** `None`
- [ ] `torch.cuda.is_available()` retourne `True`
- [ ] `torch.cuda.device_count()` retourne `2` (ou au moins `1`)
- [ ] Le script `rocm_diagnostic.py` détecte les GPUs
- [ ] L'API retourne `"device": "cuda:0"` au lieu de `"cpu"` dans `/health`

---

## 🆘 Problèmes courants

### Problème : "No module named 'torch'"
**Solution** : Vérifiez que vous êtes dans le bon environnement virtuel et que l'installation s'est bien passée.

### Problème : Installation échoue avec erreur de dépendances
**Solution** : 
```bash
# Installer les dépendances système d'abord
sudo apt update
sudo apt install -y rocm-dev rocm-libs rocm-utils
```

### Problème : `torch.version.hip` est toujours `None` après installation
**Solution** : 
1. Vérifiez que vous avez installé depuis le bon index (rocm5.7, rocm5.6, etc.)
2. Vérifiez que vous n'avez pas plusieurs installations de PyTorch qui se chevauchent
3. Essayez de réinstaller dans un environnement virtuel frais

### Problème : `cuda.is_available()` est `True` mais `device_count()` est `0`
**Solution** : 
1. Définir `HSA_OVERRIDE_GFX_VERSION=9.0.0` (pour Vega 64)
2. Vérifier les permissions `/dev/kfd` et `/dev/dri`
3. Vérifier que l'utilisateur est dans les groupes `render` et `video`

---

## 🔗 Ressources

- [PyTorch avec ROCm - Installation officielle](https://pytorch.org/get-started/locally/)
- [Index des wheels PyTorch pour ROCm](https://download.pytorch.org/whl/rocm5.7/)
- [Documentation ROCm](https://rocm.docs.amd.com/)

