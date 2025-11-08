#!/usr/bin/env python3
"""
Script de diagnostic pour vérifier la configuration ROCm et la détection des GPUs AMD.
"""
import os
import sys
import torch
import time
from pathlib import Path

print("="*70)
print("=== PyTorch ROCm Diagnostic ===")
print("="*70)

# Vérifier les variables d'environnement
print("\n📋 Variables d'environnement:")
print(f"  HSA_OVERRIDE_GFX_VERSION: {os.environ.get('HSA_OVERRIDE_GFX_VERSION', 'NON DÉFINI')}")
print(f"  ROCM_PATH: {os.environ.get('ROCM_PATH', 'NON DÉFINI')}")
print(f"  HIP_PATH: {os.environ.get('HIP_PATH', 'NON DÉFINI')}")
print(f"  HIP_VISIBLE_DEVICES: {os.environ.get('HIP_VISIBLE_DEVICES', 'NON DÉFINI (tous visibles)')}")

# Vérifier PyTorch
print("\n🔍 Informations PyTorch:")
print(f"  Version PyTorch: {torch.__version__}")

try:
    hip_version = torch.version.hip
    print(f"  Version HIP: {hip_version}")
    print(f"  ROCm disponible: ✅ Oui")
except AttributeError:
    print(f"  Version HIP: ❌ Non disponible")
    print(f"  ROCm disponible: ❌ Non (PyTorch n'a pas été compilé avec ROCm)")

print(f"  torch.cuda.is_available(): {torch.cuda.is_available()}")

# Vérifier les GPUs
if torch.cuda.is_available():
    ngpu = torch.cuda.device_count()
    print(f"\n🎮 GPUs détectés: {ngpu}")
    
    if ngpu == 0:
        print("\n⚠️  ATTENTION: torch.cuda.is_available() est True mais aucun GPU n'est détecté!")
        print("\n💡 Solutions possibles:")
        print("  1. Vérifier que HSA_OVERRIDE_GFX_VERSION est défini (ex: 9.0.0 pour Vega 64)")
        print("  2. Vérifier les permissions: ls -la /dev/kfd et ls -la /dev/dri/")
        print("  3. Vérifier que l'utilisateur est dans les groupes 'render' et 'video'")
        print("  4. Vérifier que ROCm est correctement installé: rocm-smi")
    else:
        for i in range(ngpu):
            try:
                props = torch.cuda.get_device_properties(i)
                print(f"\n  GPU {i}:")
                print(f"    Nom: {props.name}")
                if hasattr(props, 'gcnArchName'):
                    print(f"    Architecture GCN: {props.gcnArchName}")
                print(f"    Mémoire totale: {props.total_memory/1024**3:.2f} Go")
                
                # Test de calcul simple
                print(f"    Test de calcul...", end=" ", flush=True)
                device = torch.device(f"cuda:{i}")
                x = torch.randn(1000, 1000, device=device)
                y = torch.randn(1000, 1000, device=device)
                torch.cuda.synchronize()
                t0 = time.time()
                z = torch.matmul(x, y)
                torch.cuda.synchronize()
                elapsed = time.time() - t0
                print(f"✅ OK ({elapsed:.3f}s)")
                print(f"    Résultat test: {z.mean().item():.6f}")
            except Exception as e:
                print(f"    ❌ Erreur lors du test: {e}")
        
        # Test plus complet sur GPU 0
        if ngpu > 0:
            print("\n🧪 Test de performance sur GPU 0:")
            device = torch.device("cuda:0")
            x = torch.randn(8192, 8192, device=device)
            y = torch.randn(8192, 8192, device=device)
            
            torch.cuda.synchronize()
            t0 = time.time()
            z = torch.matmul(x, y)
            torch.cuda.synchronize()
            elapsed = time.time() - t0
            print(f"  Matrice 8192x8192: {elapsed:.3f} secondes")
            print(f"  Résultat moyen: {z.mean().item():.6f}")
else:
    print("\n❌ Aucun GPU ROCm utilisable détecté par PyTorch.")
    print("\n💡 Checklist de diagnostic:")
    print("  1. PyTorch a-t-il été installé avec support ROCm?")
    print("     → pip install torch --index-url https://download.pytorch.org/whl/rocm5.7")
    print("  2. ROCm est-il installé? (vérifier /opt/rocm)")
    print("  3. HSA_OVERRIDE_GFX_VERSION est-il défini? (ex: export HSA_OVERRIDE_GFX_VERSION=9.0.0)")
    print("  4. Les bibliothèques ROCm sont-elles accessibles?")
    print("     → ldconfig -p | grep rocm")
    print("  5. Les GPUs sont-ils visibles?")
    print("     → rocm-smi ou lspci | grep -i amd")

# Vérifier les fichiers système
print("\n📁 Vérification des fichiers système:")
kfd_exists = Path("/dev/kfd").exists()
dri_exists = Path("/dev/dri").exists()
print(f"  /dev/kfd existe: {'✅ Oui' if kfd_exists else '❌ Non'}")
print(f"  /dev/dri existe: {'✅ Oui' if dri_exists else '❌ Non'}")

if kfd_exists:
    try:
        kfd_stat = os.stat("/dev/kfd")
        print(f"  Permissions /dev/kfd: {oct(kfd_stat.st_mode)[-3:]}")
    except Exception as e:
        print(f"  Erreur lecture /dev/kfd: {e}")

print("\n" + "="*70)
print("Pour plus d'informations, consultez doc/DIAGNOSTIC_ROCM.md")
print("="*70)
