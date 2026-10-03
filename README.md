# AIoT Smart Locker — Face ID

**RET503 · Pertemuan 3 · Ester Aprilyani Pandiangan · 4222401026**
**Program Studi D4 Teknologi Rekayasa Robotika · Politeknik Negeri Batam**

---

## 1. Deskripsi Proyek

Proyek ini merupakan pengembangan modul Face ID untuk sistem **AIoT Smart Locker** yang dirancang untuk membantu proses autentikasi pengguna melalui pengenalan wajah. Sistem mengklasifikasikan gambar wajah pengguna terdaftar sebagai bagian dari proses akses smart locker.

Eksperimen ini membandingkan tiga arsitektur deep learning, yaitu **ResNet-18, ResNet-50, dan EfficientNet-B0**, untuk mengetahui performa klasifikasi wajah berdasarkan akurasi validation, waktu training, epoch saat mencapai akurasi tertentu, dan inference latency pada CPU.

Setiap anggota kelompok menguji arsitektur yang berbeda:

* **Asra Devi Fanitya** — ResNet-18
* **Ester Aprilyani Pandiangan** — ResNet-50
* **Hertasia Sicilia** — EfficientNet-B0

Eksperimen ini berfokus pada klasifikasi dua kelas wajah terdaftar, yaitu Asra dan Ester. Integrasi model dengan server dan perangkat fisik smart locker merupakan bagian pengembangan sistem selanjutnya.

---

## 2. Tujuan

* Mengembangkan model klasifikasi wajah menggunakan deep learning.
* Membandingkan performa ResNet-18, ResNet-50, dan EfficientNet-B0.
* Mengevaluasi pengaruh metode training terhadap hasil klasifikasi ResNet-50.
* Mengukur inference latency dan estimasi FPS pada CPU.
* Menyediakan hasil eksperimen sebagai dasar pengembangan modul Face ID untuk AIoT Smart Locker.

---

## 3. Dataset

Dataset utama terdiri dari 100 gambar wajah dari dua kelas pengguna.

| Kelas     | Jumlah Gambar |
| --------- | ------------: |
| Asra      |            50 |
| Ester     |            50 |
| **Total** |       **100** |

Pada `metadata.csv` juga terdapat data `ali` sebanyak 3 gambar dan `unknown` sebanyak 1 gambar. Data tersebut tidak digunakan dalam eksperimen utama dua kelas.

### Pembagian Dataset

Dataset Asra dan Ester berasal dari satu sesi pengambilan gambar (`sesi1`), sehingga pembagian berdasarkan sesi belum dapat dilakukan. Dataset dibagi menggunakan random split 80:20 dengan seed 42.

| Dataset    |   Asra |  Ester |   Total |
| ---------- | -----: | -----: | ------: |
| Training   |     40 |     40 |      80 |
| Validation |     10 |     10 |      20 |
| **Total**  | **50** | **50** | **100** |

Pembagian ini digunakan untuk eksperimen klasifikasi dua kelas. Karena gambar training dan validation berasal dari sesi yang sama, hasil validation belum menunjukkan kemampuan generalisasi terhadap sesi pengambilan atau kondisi kamera yang berbeda.

Dataset wajah merupakan data biometrik, sehingga foto asli dan dataset hasil pembagian tidak disertakan dalam repository publik.

---

## 4. Pipeline Preprocessing

Pipeline preprocessing yang digunakan pada eksperimen ResNet-50:

```text
Gambar
   ↓
Resize dan augmentasi saat training
   ↓
Random Horizontal Flip
   ↓
Color Jitter
   ↓
Resize / Center Crop 224 × 224 saat validation
   ↓
ToTensor
   ↓
Normalize ImageNet
   ↓
Model klasifikasi
```

Normalisasi menggunakan parameter ImageNet:

```python
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

Pada validation, gambar diubah ukurannya menjadi 256 × 256, kemudian dilakukan center crop menjadi 224 × 224 piksel. Augmentasi training menggunakan random horizontal flip dan color jitter.

---

## 5. Arsitektur Model

Tiga arsitektur yang dievaluasi oleh anggota kelompok adalah:

| Model           | Penanggung Jawab           | Keterangan                             |
| --------------- | -------------------------- | -------------------------------------- |
| ResNet-18       | Asra Devi Fanitya          | Residual network dengan 18 layer       |
| ResNet-50       | Ester Aprilyani Pandiangan | Residual network dengan 50 layer       |
| EfficientNet-B0 | Hertasia Sicilia           | Efficient convolutional neural network |

Eksperimen ResNet-18 dan EfficientNet-B0 menggunakan bobot pretrained ImageNet sesuai implementasi masing-masing anggota. Pada ResNet-50 dilakukan pengujian menggunakan tiga metode training, yaitu feature extraction, partial fine-tuning, dan scratch.

### Parameter ResNet-50

| Parameter     | Nilai                              |
| ------------- | ---------------------------------- |
| Model         | ResNet-50                          |
| Pretrained    | ImageNet untuk feature dan partial |
| Jumlah kelas  | 2                                  |
| Batch size    | 16                                 |
| Epoch         | 10                                 |
| Optimizer     | Adam                               |
| Scheduler     | CosineAnnealingLR                  |
| Loss function | Cross Entropy Loss                 |
| Device        | CPU                                |

---

## 6. Metode Training ResNet-50

Tiga metode training diuji untuk mengetahui pengaruh strategi pembelajaran terhadap performa model.

| Metode              | Penjelasan                                                                                  |
| ------------------- | ------------------------------------------------------------------------------------------- |
| Feature Extraction  | Menggunakan bobot pretrained ImageNet, backbone dibekukan dan hanya classifier yang dilatih |
| Partial Fine-Tuning | Menggunakan bobot pretrained ImageNet, layer terakhir (`layer4`) dan classifier dilatih     |
| Scratch             | Model dilatih tanpa bobot pretrained ImageNet                                               |

---

## 7. Hasil Eksperimen ResNet-50

### Hasil Training

| Parameter                   |    Feature |    Partial |    Scratch |
| --------------------------- | ---------: | ---------: | ---------: |
| Best validation accuracy    |        95% |       100% |       100% |
| Epoch pertama mencapai ≥90% |          2 |          2 |          7 |
| Waktu training (perkiraan)  | ±1,0 menit | ±1,2 menit | ±3,0 menit |
| Device                      |        CPU |        CPU |        CPU |

Hasil menunjukkan bahwa ketiga metode menghasilkan performa validation yang berbeda. Feature extraction mencapai akurasi validation terbaik sebesar 95%, sedangkan partial fine-tuning dan scratch mencapai 100% pada validation set eksperimen ini.

Akurasi 100% hanya menggambarkan hasil pada 20 gambar validation yang digunakan. Hasil ini belum dapat dianggap sebagai jaminan performa untuk data wajah baru atau kondisi pengambilan yang berbeda.

### Grafik Accuracy per Epoch

#### ResNet-50 — Feature Extraction

![ResNet-50 Feature Extraction Accuracy](results/acc_resnet50_feature.png)

#### ResNet-50 — Partial Fine-Tuning

![ResNet-50 Partial Fine-Tuning Accuracy](results/acc_resnet50_partial.png)

#### ResNet-50 — Scratch

![ResNet-50 Scratch Accuracy](results/acc_resnet50_scratch.png)

Grafik memperlihatkan perubahan validation accuracy selama proses training untuk setiap metode.

---

## 8. Pengukuran Inference Latency ResNet-50

Pengukuran inference latency dilakukan menggunakan CPU. Setiap model diuji dengan 10 iterasi warm-up dan 100 iterasi pengukuran, lalu pengujian diulang sebanyak tiga kali.

Input berupa tensor berukuran 224 × 224 piksel. Pengukuran ini hanya mencakup inferensi model, tidak termasuk pengambilan gambar dari kamera dan preprocessing.

### Hasil Pengukuran

| Mode Training       | Rata-rata Latency | Perkiraan FPS |
| ------------------- | ----------------: | ------------: |
| Feature Extraction  |          46,98 ms |         21,29 |
| Partial Fine-Tuning |          62,66 ms |         15,96 |
| Scratch             |          68,63 ms |         14,57 |

FPS dihitung menggunakan rumus:

```text
FPS = 1000 / Average Latency (ms)
```

Nilai latency dapat berubah karena beban CPU, proses lain yang berjalan, dan kondisi sistem saat pengujian. Ketiga model memiliki arsitektur ResNet-50 yang sama, sehingga perbedaan latency tidak dapat langsung dianggap sebagai perbedaan kompleksitas arsitektur.

File hasil pengukuran terakhir disimpan di `results/latency_resnet50.csv`. Tabel di atas merupakan rekapitulasi rata-rata dari tiga kali pengukuran.

---

## 9. Perbandingan Hasil Eksperimen Kelompok

Tabel berikut merangkum hasil eksperimen yang dilaporkan oleh masing-masing anggota kelompok.

| Parameter                | ResNet-18 (Asra) | ResNet-50 Feature (Ester) | ResNet-50 Partial (Ester) | ResNet-50 Scratch (Ester) | EfficientNet-B0 (Herta) |
| ------------------------ | ---------------: | ------------------------: | ------------------------: | ------------------------: | ----------------------: |
| Best validation accuracy |              95% |                       95% |                      100% |                      100% |                    100% |
| Epoch pertama ≥90%       |                4 |                         2 |                         2 |                         7 |                       — |
| Epoch terbaik            |                6 |                         — |                         — |                         — |                       1 |
| Waktu training           |       ±1,0 menit |                ±1,0 menit |                ±1,2 menit |                ±3,0 menit |            141,64 detik |
| Average latency          |         59,92 ms |                  46,98 ms |                  62,66 ms |                  68,63 ms |                56,40 ms |
| Perkiraan FPS            |             16,7 |                     21,29 |                     15,96 |                     14,57 |                   17,73 |

**Catatan perbandingan:**

* ResNet-18 dan EfficientNet-B0 merupakan hasil eksperimen masing-masing anggota, sedangkan tiga baris ResNet-50 merupakan tiga metode training dari anggota yang sama.
* Informasi epoch pertama mencapai ≥90% untuk EfficientNet-B0 tidak dicantumkan karena tidak tersedia pada hasil yang diberikan.
* Waktu dan pengaturan pengujian masing-masing anggota dapat berbeda. Oleh karena itu, tabel ini merupakan ringkasan hasil eksperimen kelompok, bukan perbandingan benchmark terkontrol sepenuhnya.
* Latency antaranggota sebaiknya diuji ulang dengan prosedur, input, perangkat, dan jumlah iterasi yang sama sebelum digunakan sebagai perbandingan final.

---

## 10. Analisis Hasil

Eksperimen ResNet-50 menunjukkan bahwa feature extraction memperoleh validation accuracy sebesar 95%, sementara partial fine-tuning dan scratch memperoleh 100%. Pada eksperimen ini, feature extraction dan partial fine-tuning pertama kali mencapai validation accuracy minimal 90% pada epoch ke-2, sedangkan scratch mencapainya pada epoch ke-7.

Hasil ResNet-18 dari Asra memperoleh validation accuracy terbaik sebesar 95% pada epoch ke-6, dengan waktu training sekitar satu menit dan average latency sebesar 59,92 ms.

EfficientNet-B0 dari Herta memperoleh validation accuracy sebesar 100% pada epoch pertama, waktu training 141,64 detik, dan average latency sebesar 56,40 ms.

Perbedaan hasil tersebut menunjukkan bahwa performa model dipengaruhi oleh arsitektur, metode training, serta konfigurasi eksperimen. Namun, karena ukuran validation set masih kecil dan pengaturan eksperimen anggota belum sepenuhnya diseragamkan, hasil ini belum cukup untuk menyimpulkan kemampuan generalisasi model pada penggunaan smart locker sesungguhnya.

---

## 11. Struktur Folder

```text
smart-locker-faceid/
├── dataset_raw/                 # Dataset wajah lokal, tidak di-upload
├── dataset_split/               # Hasil split lokal, tidak di-upload
├── results/
│   ├── acc_resnet50_feature.png
│   ├── acc_resnet50_partial.png
│   ├── acc_resnet50_scratch.png
│   ├── log_resnet50_feature.csv
│   ├── log_resnet50_partial.csv
│   ├── log_resnet50_scratch.csv
│   ├── summary.csv
│   └── latency_resnet50.csv
├── scripts/
│   ├── capture.py
│   ├── split.py
│   ├── train.py
│   └── latency.py
├── .gitignore
└── README.md
```

Checkpoint model (`*.pth`) dan foto wajah tidak disertakan dalam repository untuk menghindari file besar dan menjaga privasi data biometrik.

---

## 12. Instalasi

Pastikan Python telah terpasang. Kemudian instal pustaka yang dibutuhkan:

```bash
python -m pip install torch torchvision
python -m pip install pillow numpy matplotlib
```

Jika repository memiliki `requirements.txt`, dependensi dapat diinstal menggunakan:

```bash
python -m pip install -r requirements.txt
```

---

## 13. Persiapan Dataset

Dataset tidak disertakan dalam repository publik. Untuk menjalankan eksperimen secara lokal, siapkan dataset sesuai struktur yang dibutuhkan oleh script:

```text
dataset_raw/
├── Asra/
└── Ester/
```

Pastikan gambar wajah hanya disimpan secara lokal dan tidak diunggah ke repository publik.

---

## 14. Menjalankan Eksperimen ResNet-50

### Membagi Dataset

```bash
python scripts/split.py
```

### Training Feature Extraction

```bash
python scripts/train.py --model resnet50 --mode feature
```

### Training Partial Fine-Tuning

```bash
python scripts/train.py --model resnet50 --mode partial
```

### Training from Scratch

```bash
python scripts/train.py --model resnet50 --mode scratch
```

### Mengukur Inference Latency

```bash
python scripts/latency.py
```

Hasil eksperimen akan disimpan di dalam folder `results/`.

---

## 15. Keterbatasan dan Pengembangan Selanjutnya

Beberapa keterbatasan eksperimen saat ini:

1. Dataset utama hanya terdiri dari 100 gambar wajah untuk dua kelas.
2. Data Asra dan Ester hanya berasal dari satu sesi pengambilan.
3. Validation set hanya terdiri dari 20 gambar.
4. Pengujian dilakukan pada CPU laptop, bukan perangkat final smart locker.
5. Belum dilakukan pengujian yang memadai terhadap wajah pengguna yang tidak terdaftar.
6. Latency yang diukur belum mencakup keseluruhan proses dari kamera hingga respons smart locker.
7. Perbandingan antaranggota belum sepenuhnya menggunakan prosedur benchmark yang sama.

Pengembangan berikutnya dapat mencakup:

* Menambah data dari beberapa sesi, sudut wajah, dan kondisi pencahayaan.
* Menambah kelas pengguna sesuai kebutuhan sistem.
* Menguji model menggunakan test set terpisah.
* Menguji wajah yang tidak terdaftar dan mengembangkan mekanisme penolakan autentikasi.
* Mengukur end-to-end latency pada perangkat target.
* Menyeragamkan prosedur eksperimen untuk membandingkan ResNet-18, ResNet-50, dan EfficientNet-B0.
* Mengintegrasikan model dengan server dan perangkat fisik smart locker.

---

## 16. Catatan Privasi

Proyek ini menggunakan data wajah yang termasuk data biometrik. Oleh karena itu, foto wajah mentah, dataset hasil pembagian, dan checkpoint model tidak disertakan dalam repository publik. Penggunaan data harus dilakukan dengan memperhatikan izin dan privasi orang yang terlibat.

---

**Program Studi D4 Teknologi Rekayasa Robotika**
**Politeknik Negeri Batam**

**AIoT Smart Locker — Face ID Module**
