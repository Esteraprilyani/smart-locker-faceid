"""
train.py - Perbandingan 3 arsitektur x 3 mode latih (sesuai materi praktikum P2)

Arsitektur : resnet18 | resnet50 | efficientnet_b0
Mode latih :
  feature -> bobot ImageNet, hanya head (fc/classifier) dilatih, lr 1e-3
  partial -> bobot ImageNet, blok terakhir + head dilatih, lr 1e-4 / 1e-3
  scratch -> bobot acak, semua dilatih, lr 1e-3

Penggunaan:
  python scripts/train.py --model resnet50 --mode feature
  python scripts/train.py --model resnet50 --mode partial
  python scripts/train.py --model resnet50 --mode scratch

Keluaran (folder results/):
  log_<model>_<mode>.csv   -> epoch, train_loss, val_acc
  best_<model>_<mode>.pth  -> bobot terbaik
  acc_<model>_<mode>.png   -> grafik akurasi per epoch
  summary.csv              -> ringkasan semua percobaan (untuk tabel hasil)
"""

import os, time, argparse, csv
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, models, transforms

BATCH  = 16
EPOCHS = 10
SEED   = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA   = 'dataset_split'


def get_transforms():
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        norm,
    ])
    val_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        norm,
    ])
    return train_tf, val_tf


def build_model(model_name, mode, num_classes):
    """Buat model + optimizer sesuai arsitektur dan mode latih."""
    pretrained = (mode != 'scratch')

    if model_name == 'resnet18':
        w = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.resnet18(weights=w)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        head, last_block = m.fc, m.layer4

    elif model_name == 'resnet50':
        w = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.resnet50(weights=w)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        head, last_block = m.fc, m.layer4

    elif model_name == 'efficientnet_b0':
        w = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.efficientnet_b0(weights=w)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        head, last_block = m.classifier[1], m.features[-2:]  # padanan layer4

    else:
        raise ValueError(f"Model tidak dikenal: {model_name}")

    if mode == 'feature':
        for p in m.parameters():
            p.requires_grad = False
        for p in head.parameters():
            p.requires_grad = True
        opt = Adam(head.parameters(), lr=1e-3)

    elif mode == 'partial':
        for p in m.parameters():
            p.requires_grad = False
        for p in last_block.parameters():
            p.requires_grad = True
        for p in head.parameters():
            p.requires_grad = True
        opt = Adam([
            {'params': last_block.parameters(), 'lr': 1e-4},
            {'params': head.parameters(),       'lr': 1e-3},
        ])

    elif mode == 'scratch':
        opt = Adam(m.parameters(), lr=1e-3)

    else:
        raise ValueError(f"Mode tidak dikenal: {mode}")

    return m.to(DEVICE), opt


def save_plot(epochs, accs, path, title):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("[INFO] matplotlib belum terpasang, grafik dilewati "
              "(pip install matplotlib)")
        return
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, accs, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Akurasi val (%)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True,
                        choices=['resnet18', 'resnet50', 'efficientnet_b0'])
    parser.add_argument('--mode', required=True,
                        choices=['feature', 'partial', 'scratch'])
    args = parser.parse_args()
    tag = f'{args.model}_{args.mode}'

    torch.manual_seed(SEED)

    train_tf, val_tf = get_transforms()
    train_ds = datasets.ImageFolder(os.path.join(DATA, 'train'), train_tf)
    val_ds   = datasets.ImageFolder(os.path.join(DATA, 'val'),   val_tf)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=0)
    val_dl   = torch.utils.data.DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=0)

    num_classes = len(train_ds.classes)
    print(f"\n[INFO] Model   : {args.model}")
    print(f"[INFO] Mode    : {args.mode}")
    print(f"[INFO] Kelas   : {train_ds.classes}")
    print(f"[INFO] Train   : {len(train_ds)} foto | Val: {len(val_ds)} foto")
    print(f"[INFO] Device  : {DEVICE}")
    print(f"[INFO] Epochs  : {EPOCHS}\n")

    model, opt = build_model(args.model, args.mode, num_classes)
    criterion  = nn.CrossEntropyLoss()
    scheduler  = CosineAnnealingLR(opt, T_max=EPOCHS)

    os.makedirs('results', exist_ok=True)
    log_path = f'results/log_{tag}.csv'
    best_acc = 0.0
    epoch_90 = None
    hist_ep, hist_acc = [], []
    t_start = time.time()

    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_acc'])

        for epoch in range(1, EPOCHS + 1):
            # -- Train --
            model.train()
            running_loss = 0.0
            for imgs, labels in train_dl:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                opt.zero_grad()
                loss = criterion(model(imgs), labels)
                loss.backward()
                opt.step()
                running_loss += loss.item() * imgs.size(0)
            train_loss = running_loss / len(train_ds)

            # -- Val --
            model.eval()
            correct = 0
            with torch.no_grad():
                for imgs, labels in val_dl:
                    imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                    correct += (model(imgs).argmax(1) == labels).sum().item()
            val_acc = correct / len(val_ds) * 100

            scheduler.step()
            writer.writerow([epoch, f'{train_loss:.4f}', f'{val_acc:.2f}'])
            f.flush()
            hist_ep.append(epoch)
            hist_acc.append(val_acc)
            print(f"Epoch {epoch:2d}/{EPOCHS} | Loss: {train_loss:.4f} | Val Acc: {val_acc:.2f}%")

            if val_acc >= 90 and epoch_90 is None:
                epoch_90 = epoch

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), f'results/best_{tag}.pth')

    elapsed = time.time() - t_start

    save_plot(hist_ep, hist_acc, f'results/acc_{tag}.png',
              f'{args.model} - {args.mode}')

    summary_path = 'results/summary.csv'
    new_file = not os.path.exists(summary_path)
    with open(summary_path, 'a', newline='') as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(['model', 'mode', 'best_val_acc', 'waktu_latih_menit', 'epoch_acc_90'])
        w.writerow([args.model, args.mode, f'{best_acc:.2f}',
                    f'{elapsed/60:.1f}', epoch_90 if epoch_90 else '-'])

    print(f"\n{'='*50}")
    print(f"Model        : {args.model}")
    print(f"Mode         : {args.mode}")
    print(f"Best Val Acc : {best_acc:.2f}%")
    print(f"Waktu Latih  : {elapsed/60:.1f} menit")
    print(f"Epoch @ 90%  : {epoch_90 if epoch_90 else 'tidak tercapai'}")
    print(f"Log          : {log_path}")
    print(f"{'='*50}\n")


if __name__ == '__main__':
    main()