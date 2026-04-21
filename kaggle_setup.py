import os

import torch

from src.models.memory_augmented_model_v2 import CultivatedMemoryModel


MODEL_ID = os.getenv("AL_NISYAN_MODEL_ID", "Qwen/Qwen3-7B")
MEMORY_SLOTS = int(os.getenv("AL_NISYAN_MEMORY_SLOTS", "1024"))
MEMORY_DIM = int(os.getenv("AL_NISYAN_MEMORY_DIM", "512"))
CONTROLLER_MODE = os.getenv("AL_NISYAN_CONTROLLER_MODE", "clean")

print("=" * 70)
print("AL-NISYAN Kaggle Setup (T4 15GB)")
print("=" * 70)
print(f"Model ID: {MODEL_ID}")
print(f"Memory: {MEMORY_SLOTS} slots x {MEMORY_DIM} dim")
print(f"Controller mode: {CONTROLLER_MODE}")

model = CultivatedMemoryModel(
    base_model_key=MODEL_ID,
    memory_slots=MEMORY_SLOTS,
    memory_dim=MEMORY_DIM,
    controller_mode=CONTROLLER_MODE,
)

if torch.cuda.is_available():
    used = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"VRAM allocated: {used:.2f} GB")
    print(f"VRAM reserved:  {reserved:.2f} GB")
else:
    print("CUDA not available; running on CPU")

print(f"Model loaded: {model.loader.get_model_info()}")
