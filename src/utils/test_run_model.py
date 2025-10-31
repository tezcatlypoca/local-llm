#!/usr/bin/env python3
import os
import argparse
import torch
import torchvision.models as models
import time

def main():
    parser = argparse.ArgumentParser(description="Configurer la taille du cache MIOpen et tester l’inférence")
    parser.add_argument("--cache-size", type=int, default=4096,
                        help="Taille maximale du cache MIOpen en Mo (défaut : 4096 Mo)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Taille du batch pour le test d’inférence (défaut : 16)")
    args = parser.parse_args()

    # Configuration du cache MIOpen
    os.environ["MIOPEN_USER_DB_PATH"] = os.path.expanduser("~/.cache/miopen")
    os.environ["MIOPEN_CACHE_DIR"] = os.path.expanduser("~/.cache/miopen")
    os.environ["MIOPEN_CACHE_MAX_SIZE"] = str(args.cache_size)
    os.environ["MIOPEN_ENABLE_CACHE"] = "1"

    print("===============================================")
    print("=== Configuration du cache MIOpen ===")
    print(f"Répertoire du cache : {os.environ['MIOPEN_CACHE_DIR']}")
    print(f"Taille maximale du cache : {args.cache_size} Mo")
    print("===============================================\n")

    # Sélection du GPU 1 (pour éviter le GPU d’affichage)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Appareil utilisé : {device}")

    # Chargement du modèle
    model = models.mobilenet_v2().to(device)
    model.eval()
    print(f"Le modèle est sur : {next(model.parameters()).device}")

    # Test d’inférence
    dummy_input = torch.randn(args.batch_size, 3, 224, 224, device=device)
    print("\nTest d’inférence en cours...")

    start = time.time()
    with torch.no_grad():
        _ = model(dummy_input)
    duration = time.time() - start

    print(f"Temps d’inférence pour un batch de {args.batch_size} images : {duration:.3f} s")
    print("Si tu relances le script une 2e fois, tu devrais observer un gain (cache MIOpen actif).")

if __name__ == "__main__":
    main()
