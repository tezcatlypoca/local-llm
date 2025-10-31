python - <<'EOF'
import torch, torchvision, torchaudio
print("Torch :", torch.__version__)
print("Torchvision :", torchvision.__version__)
print("Torchaudio :", torchaudio.__version__)
print("GPU dispo :", torch.cuda.is_available())
print("GPU utilisé :", torch.cuda.get_device_name(0))
EOF
