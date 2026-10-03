
"""
Mengukur inference latency model ResNet-50 hasil training.
Mode: feature, partial, scratch
Pengukuran dilakukan di CPU, untuk satu gambar berukuran 224x224.
"""

import os
import time
import csv
import torch
import torch.nn as nn
from torchvision import models

DEVICE = torch.device("cpu")
N_WARMUP = 10
N_RUNS = 100
NUM_CLASSES = 2
IMG_SIZE = 224

CHECKPOINTS = {
    "feature": "results/best_resnet50_feature.pth",
    "partial": "results/best_resnet50_partial.pth",
    "scratch": "results/best_resnet50_scratch.pth",
}


def load_model(checkpoint_path):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    state_dict = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=True
    )
    model.load_state_dict(state_dict)
    model.eval()
    model.to(DEVICE)

    return model


def measure_latency(model):
    # Tensor tiruan untuk mengukur waktu inferensi model saja.
    # Resize dan preprocessing tidak termasuk dalam pengukuran.
    tensor = torch.rand(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)

    with torch.inference_mode():
        # Warm-up agar pengukuran lebih stabil
        for _ in range(N_WARMUP):
            model(tensor)

        times = []
        for _ in range(N_RUNS):
            start = time.perf_counter()
            model(tensor)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

    avg = sum(times) / len(times)
    return avg, min(times), max(times), 1000 / avg


def main():
    os.makedirs("results", exist_ok=True)
    output_csv = "results/latency_resnet50.csv"
    results = []

    print("\nPengukuran Inference Latency ResNet-50")
    print(f"Device: {DEVICE} | Warm-up: {N_WARMUP} | Runs: {N_RUNS}")
    print("-" * 72)
    print(f"{'Mode':<12}{'Avg (ms)':>12}{'Min (ms)':>12}"
          f"{'Max (ms)':>12}{'FPS':>12}")

    for mode, path in CHECKPOINTS.items():
        if not os.path.exists(path):
            print(f"File tidak ditemukan: {path}")
            continue

        model = load_model(path)
        avg, minimum, maximum, fps = measure_latency(model)

        print(f"{mode:<12}{avg:>12.2f}{minimum:>12.2f}"
              f"{maximum:>12.2f}{fps:>12.2f}")

        results.append([mode, avg, minimum, maximum, fps])
        del model

    with open(output_csv, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["mode", "avg_ms", "min_ms", "max_ms", "fps"])
        writer.writerows(results)

    print("-" * 72)
    print(f"Hasil disimpan di: {output_csv}")


if __name__ == "__main__":
    main()