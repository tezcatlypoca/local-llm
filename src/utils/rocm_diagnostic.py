import torch, time

print("="*60)
print("=== PyTorch ROCm Diagnostic ===")
print("="*60)

print(f"PyTorch version: {torch.__version__}")
print(f"HIP version: {torch.version.hip}")
print(f"ROCm available: {torch.version.hip is not None}")
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")

if torch.cuda.is_available():
    ngpu = torch.cuda.device_count()
    print(f"\nNombre de GPU(s) détectés: {ngpu}")
    for i in range(ngpu):
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}: {props.name}")
        print(f" - GCN arch: {props.gcnArchName}")
        print(f" - Mémoire totale: {props.total_memory/1024**3:.2f} Go")

    # Petit test de calcul
    device = torch.device("cuda:0")
    print("\n=== Test de calcul sur GPU ===")
    x = torch.randn(8192, 8192, device=device)
    y = torch.randn(8192, 8192, device=device)

    torch.cuda.synchronize()
    t0 = time.time()
    z = torch.matmul(x, y)
    torch.cuda.synchronize()
    print(f"Calcul effectué sur {device} en {time.time() - t0:.3f} secondes.")
    print(f"Résultat moyen: {z.mean().item():.6f}")
else:
    print("⚠️ Aucun GPU ROCm utilisable détecté par PyTorch.")
