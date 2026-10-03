# AIoT Smart Locker — ResNet-50 Face ID

**RET503 · Pertemuan 3 · Ester Aprilyani Pandiangan · 4222401026**

## Deskripsi Proyek

Proyek ini merupakan eksperimen modul Face ID untuk sistem AIoT Smart Locker menggunakan model **ResNet-50**. Model digunakan untuk mengklasifikasikan wajah pengguna terdaftar ke dalam dua kelas, yaitu **Asra** dan **Ester**.

Eksperimen ini bertujuan untuk membandingkan performa ResNet-50 dengan tiga pendekatan training, yaitu **Feature Extraction, Partial Fine-Tuning, dan Training from Scratch**.

## Dataset

Dataset utama terdiri dari 100 gambar wajah dari dua kelas.

| Kelas     |  Jumlah |
| --------- | ------: |
| Asra      |      50 |
| Ester     |      50 |
| **Total** | **100** |

### Pembagian Dataset

Dataset dibagi menggunakan stratified random split dengan rasio 80:20.

| Dataset    |   Asra |  Ester |   Total |
| ---------- | -----: | -----: | ------: |
| Training   |     40 |     40 |      80 |
| Validation |     10 |     10 |      20 |
| **Total**  | **50** | **50** | **100** |

Dataset wajah tidak disertakan dalam repository untuk menjaga privasi.

## Model ResNet-50

Eksperimen menggunakan arsitektur **ResNet-50** untuk klasifikasi dua kelas wajah terdaftar.

Tiga pendekatan training yang diuji:

* **Feature Extraction:** menggunakan backbone pretrained sebagai pengekstraksi fitur dan melatih classifier.
* **Partial Fine-Tuning:** melatih classifier dan sebagian layer backbone.
* **Training from Scratch:** melatih model tanpa menggunakan bobot pretrained.

## Metode Eksperimen

Ketiga mode training dijalankan dan dievaluasi menggunakan dataset training dan validation yang sama agar hasilnya dapat dibandingkan.

Parameter dan konfigurasi training mengikuti implementasi pada `scripts/train.py`.

## Hasil Eksperimen

| Mode Training         | Akurasi Validation Terbaik | Waktu Training |
| --------------------- | -------------------------: | -------------: |
| Feature Extraction    |                        95% |      1,0 menit |
| Partial Fine-Tuning   |                       100% |      1,2 menit |
| Training from Scratch |                       100% |      3,0 menit |

### Grafik Accuracy

**Feature Extraction**

![ResNet-50 Feature Extraction](results/acc_resnet50_feature.png)

**Partial Fine-Tuning**

![ResNet-50 Partial Fine-Tuning](results/acc_resnet50_partial.png)

**Training from Scratch**

![ResNet-50 Scratch](results/acc_resnet50_scratch.png)

## Analisis Hasil

Berdasarkan eksperimen yang dilakukan, ResNet-50 dengan mode Feature Extraction memperoleh akurasi validation terbaik sebesar 95%, sedangkan Partial Fine-Tuning dan Training from Scratch memperoleh akurasi validation terbaik sebesar 100%.

Waktu training berbeda pada setiap pendekatan. Feature Extraction membutuhkan sekitar 1,0 menit, Partial Fine-Tuning sekitar 1,2 menit, dan Training from Scratch sekitar 3,0 menit.

Hasil ini menunjukkan performa model pada dataset validation dalam eksperimen yang dilakukan. Namun, akurasi yang tinggi belum menjamin kemampuan generalisasi pada kondisi nyata, karena jumlah dataset masih terbatas dan data validation berasal dari pembagian dataset yang sama.

## Keterbatasan Eksperimen

Beberapa keterbatasan eksperimen ini adalah:

* Dataset hanya terdiri dari 100 gambar wajah.
* Eksperimen menggunakan dua kelas wajah.
* Data validation masih terbatas, yaitu 20 gambar.
* Pengujian belum mencakup berbagai sesi pengambilan gambar dan kondisi lingkungan yang beragam.
* Pengujian pada dataset terbatas belum cukup untuk memastikan keandalan Face ID pada penggunaan nyata.

Pengembangan selanjutnya dapat dilakukan dengan menambah jumlah gambar, menggunakan beberapa sesi pengambilan data, serta menguji model dalam kondisi pencahayaan dan posisi wajah yang lebih bervariasi.

## Struktur Folder

```text
smart-locker-faceid/
├── scripts/
│   └── train.py
├── results/
│   ├── acc_resnet50_feature.png
│   ├── acc_resnet50_partial.png
│   ├── acc_resnet50_scratch.png
│   ├── log_resnet50_feature.csv
│   ├── log_resnet50_partial.csv
│   ├── log_resnet50_scratch.csv
│   └── summary.csv
├── dataset_raw/          # Dataset lokal, tidak diunggah
├── .gitignore
└── README.md
```

File model hasil training (`.pth`) tidak disertakan dalam repository karena ukuran file yang besar. Dataset foto wajah juga tetap disimpan secara lokal untuk menjaga privasi.

## Catatan

Eksperimen ini berfokus pada perbandingan tiga pendekatan training ResNet-50 sebagai bagian dari pengembangan modul Face ID pada sistem AIoT Smart Locker.

**Program Studi Teknologi Rekayasa Robotika · Politeknik Negeri Batam**
