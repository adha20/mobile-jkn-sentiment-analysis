# Analisis Sentimen Review Mobile JKN

Repositori ini berisi pipeline analisis sentimen untuk review aplikasi **Mobile JKN** di Google Play. Target klasifikasinya terdiri dari tiga kelas: `negatif`, `netral`, dan `positif`.

Dataset dikumpulkan dengan scraping, lalu diberi label menggunakan pendekatan **weak supervision** berbasis lexicon. Model utamanya adalah **Multi-Layer Perceptron (MLP)** dengan fitur gabungan **TF-IDF**, **TruncatedSVD**, dan fitur lexicon.

## Ringkasan

Alur utama yang dikerjakan:

1. Mengambil review aplikasi Mobile JKN dari Google Play.
2. Membersihkan dan menyusun dataset.
3. Membuat label sentimen otomatis dengan weak supervision.
4. Membangun fitur teks menggunakan TF-IDF dan lexicon.
5. Melatih 3 skema model deep learning.
6. Mengevaluasi model dengan akurasi, classification report, dan confusion matrix.
7. Menyediakan contoh inference untuk teks baru.

## Dataset

Dataset berasal dari review publik aplikasi **Mobile JKN** di Google Play.

Detail dataset:

| Item | Nilai |
|---|---:|
| Sumber | Google Play Store |
| Aplikasi | Mobile JKN |
| App ID | `app.bpjs.mobile` |
| Format | CSV |
| Jumlah data | 12.000 review |
| Jumlah kelas | 3 kelas |
| Bahasa | Indonesia |

Kolom utama dataset:

| Kolom | Deskripsi |
|---|---|
| `review_id` | ID unik review |
| `content` | Isi review pengguna |
| `score` | Rating bintang dari Google Play |
| `thumbs_up_count` | Jumlah like pada review |
| `at` | Tanggal review |
| `app_version` | Versi aplikasi saat review dibuat |
| `rating_sentiment` | Label sentimen dari rating bintang |
| `sentiment` | Label training hasil weak supervision dari teks |
| `source_app` | Nama aplikasi |
| `source_app_id` | Package ID aplikasi |
| `scrape_country` | Negara scraping |
| `scrape_language` | Bahasa scraping |

Distribusi `rating_sentiment` dibuat seimbang:

| Rating Sentiment | Jumlah |
|---|---:|
| negatif | 4.000 |
| netral | 4.000 |
| positif | 4.000 |

Distribusi label training `sentiment`:

| Sentiment | Jumlah |
|---|---:|
| negatif | 6.782 |
| netral | 2.805 |
| positif | 2.413 |

## Weak Supervision

Review Google Play tidak menyediakan label sentimen manual. Karena itu, label dibuat dengan **weak supervision**, yaitu pelabelan otomatis menggunakan aturan sederhana.

Ada dua jenis label yang disimpan:

1. `rating_sentiment`

   Label berdasarkan rating:

   - Rating 1-2 -> `negatif`
   - Rating 3 -> `netral`
   - Rating 4-5 -> `positif`

2. `sentiment`

   Label berbasis isi teks review. Script menghitung kemunculan kata positif dan negatif dari lexicon sederhana.

   Contoh kata positif:

   - `bagus`
   - `mudah`
   - `membantu`
   - `lancar`
   - `cepat`

   Contoh kata negatif:

   - `error`
   - `gagal`
   - `susah`
   - `tidak bisa`
   - `lemot`

Aturan label teks:

| Kondisi | Label |
|---|---|
| Kata negatif lebih banyak | `negatif` |
| Kata positif lebih banyak | `positif` |
| Seimbang atau tidak ditemukan kata kunci | `netral` |

Catatan penting: weak supervision bukan pengganti anotasi manusia. Metrik di sini menunjukkan seberapa baik model mengikuti label otomatis, bukan seberapa sempurna model memahami sentimen menurut penilaian manusia.

## Metodologi

Alur kerja:

```text
Scraping review
-> Pembersihan data
-> Weak labeling
-> Preprocessing teks
-> Feature engineering
-> Training model MLP
-> Evaluasi
-> Inference
```

Tahapannya:

1. **Scraping**

   Review diambil menggunakan package `google-play-scraper`.

2. **Data cleaning**

   Review kosong dihapus, kolom dipilih, nama kolom dirapikan, dan duplikasi berdasarkan `review_id` dibuang.

3. **Labeling**

   Label rating disimpan sebagai `rating_sentiment`, sedangkan label training dibuat sebagai `sentiment`.

4. **Feature engineering**

   Teks diubah menjadi fitur numerik menggunakan TF-IDF, TruncatedSVD, dan fitur lexicon.

5. **Modeling**

   Model MLP dilatih dengan PyTorch.

6. **Evaluation**

   Model dievaluasi menggunakan training accuracy, testing accuracy, classification report, dan confusion matrix.

7. **Inference**

   Notebook menyediakan contoh prediksi sentimen untuk beberapa teks baru.

## Feature Engineering

Fitur model dibuat dari kombinasi representasi teks dan sinyal sederhana dari lexicon.

### Text Cleaning

Teks review dibersihkan dengan beberapa langkah:

- lowercase
- menghapus URL
- menghapus karakter non-alfanumerik
- merapikan spasi

### Word TF-IDF

Word TF-IDF menangkap pola berbasis kata:

```python
ngram_range=(1, 2)
```

Dengan konfigurasi ini, model membaca unigram dan bigram, misalnya:

- `error`
- `tidak bisa`
- `sangat membantu`

### Character TF-IDF

Character TF-IDF menangkap pola berbasis karakter:

```python
ngram_range=(3, 5)
```

Fitur ini berguna untuk menangkap variasi penulisan seperti:

- `error`
- `eror`
- `gabisa`
- `gak bisa`

### TruncatedSVD

TF-IDF menghasilkan fitur berdimensi besar. `TruncatedSVD` dipakai untuk mereduksi fitur menjadi representasi yang lebih ringkas sebelum masuk ke MLP.

### Lexicon Features

Selain TF-IDF, model memakai beberapa fitur sederhana dari lexicon:

- jumlah kata positif
- jumlah kata negatif
- jumlah kata netral
- selisih jumlah kata positif dan negatif
- indikator ada/tidaknya kata positif
- indikator ada/tidaknya kata negatif
- indikator ada/tidaknya kata netral
- panjang teks
- jumlah kata

### StandardScaler

Fitur dinormalisasi dengan `StandardScaler` supaya training neural network lebih stabil.

## Model

Modelnya adalah **TextMLP**, neural network sederhana berbasis beberapa fully connected layer.

Komponen model:

- `Linear`
- `ReLU`
- `BatchNorm1d`
- `Dropout`
- output layer untuk 3 kelas sentimen

Training menggunakan:

| Komponen | Nilai |
|---|---|
| Framework | PyTorch |
| Optimizer | AdamW |
| Loss function | CrossEntropyLoss |
| Epoch | 25 |
| Batch size | 256 |
| Random seed | 42 |

## Hasil Eksperimen

Tiga skema pelatihan dicoba dengan kombinasi fitur dan pembagian data yang berbeda.

| Skema | Feature Mode | Split | Train Samples | Test Samples | Train Accuracy | Test Accuracy |
|---|---|---:|---:|---:|---:|---:|
| MLP + Word TF-IDF + Lexicon | `word` | 80/20 | 9.600 | 2.400 | 99,97% | 99,83% |
| MLP + Char TF-IDF + Lexicon | `char` | 80/20 | 9.600 | 2.400 | 99,99% | 100,00% |
| MLP + Word-Char TF-IDF + Lexicon | `word_char` | 70/30 | 8.400 | 3.600 | 100,00% | 99,97% |

Semua skema melewati target akurasi 92% pada training set dan testing set.

## Contoh Prediksi

Contoh inference yang ada di notebook:

| Review | Prediksi |
|---|---|
| Aplikasinya mudah digunakan dan sangat membantu untuk cek jadwal berobat. | `positif` |
| Tidak bisa login, verifikasi wajah selalu gagal dan aplikasinya error terus. | `negatif` |
| Baru download, semoga aplikasinya bisa dipakai dengan baik. | `netral` |
| Pelayanan cepat, fitur antrean online lancar dan praktis. | `positif` |
| Update terbaru malah lemot, sering crash saat mau ambil antrean. | `negatif` |

## Struktur Folder

```text
submission_mobile_jkn/
|
|-- NLP_Mobile_JKN_Sentiment_Analysis.ipynb
|-- scrape_mobile_jkn_reviews.py
|-- requirements.txt
|-- README.md
|
|-- data/
|   |-- mobile_jkn_reviews.csv
|
|-- models/
|   |-- experiment_metrics.csv
|   |-- mobile_jkn_mlp_state.pt
|   |-- mobile_jkn_preprocessing.joblib
```

Penjelasan file:

| File | Deskripsi |
|---|---|
| `scrape_mobile_jkn_reviews.py` | Script scraping dan pembuatan dataset |
| `NLP_Mobile_JKN_Sentiment_Analysis.ipynb` | Notebook training, evaluasi, dan inference |
| `requirements.txt` | Daftar dependency |
| `data/mobile_jkn_reviews.csv` | Dataset hasil scraping |
| `models/experiment_metrics.csv` | Ringkasan hasil eksperimen |
| `models/mobile_jkn_mlp_state.pt` | Bobot model PyTorch terbaik |
| `models/mobile_jkn_preprocessing.joblib` | Vectorizer, SVD, scaler, label encoder, dan konfigurasi preprocessing |

## Cara Menjalankan Proyek

### 1. Clone repository

```bash
git clone https://github.com/adha20/mobile-jkn-sentiment-analysis.git
cd nama-repository
```

Jika repository langsung berisi folder proyek:

```bash
cd submission_mobile_jkn
```

### 2. Buat virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependency

```bash
pip install -r requirements.txt
```

### 4. Scraping data

Jalankan script berikut untuk mengambil ulang data review:

```bash
python scrape_mobile_jkn_reviews.py --target-per-class 4000 --output data/mobile_jkn_reviews.csv
```

Parameter opsional:

```bash
python scrape_mobile_jkn_reviews.py \
  --target-per-class 4000 \
  --batch-size 200 \
  --sleep 0.2 \
  --overfetch-ratio 1.15 \
  --output data/mobile_jkn_reviews.csv
```

Catatan: scraping membutuhkan koneksi internet. Hasilnya bisa berubah karena review Google Play terus bertambah.

### 5. Training model

Buka notebook:

```bash
jupyter notebook NLP_Mobile_JKN_Sentiment_Analysis.ipynb
```

Lalu jalankan semua cell dari atas ke bawah.

Alternatif melalui terminal:

```bash
jupyter nbconvert --to notebook --execute --inplace NLP_Mobile_JKN_Sentiment_Analysis.ipynb
```

### 6. Melihat hasil eksperimen

Hasil eksperimen tersimpan di:

```text
models/experiment_metrics.csv
```

## Teknologi yang Digunakan

| Teknologi | Kegunaan |
|---|---|
| Python | Bahasa utama |
| google-play-scraper | Scraping review Google Play |
| Pandas | Pengolahan dataset |
| NumPy | Operasi numerik |
| Scikit-learn | TF-IDF, SVD, scaler, split data, evaluasi |
| PyTorch | Deep learning model |
| Joblib | Menyimpan preprocessing artifact |
| Jupyter Notebook | Eksperimen dan dokumentasi training |

## Catatan dan Batasan

Beberapa catatan penting:

- Label `sentiment` dibuat otomatis dengan weak supervision, bukan anotasi manual manusia.
- Akurasi yang tinggi menunjukkan model sangat baik mengikuti pola label otomatis.
- Untuk penggunaan produksi, sebagian data sebaiknya divalidasi manual.
- Lexicon masih bisa diperluas agar lebih tahan terhadap slang, typo, singkatan, dan sarkasme.
- Model bisa dikembangkan lagi menggunakan pretrained language model seperti IndoBERT.
- Dataset review bisa berubah jika scraping dijalankan ulang di waktu berbeda.

## Pengembangan Selanjutnya

Beberapa langkah lanjutan yang paling relevan:

### 1. Error Analysis

Menganalisis sampel yang salah klasifikasi untuk melihat pola kesalahan model.

Hal yang perlu diamati antara lain:

- kalimat dengan negasi, seperti `tidak bagus`, `kurang membantu`, atau `belum bisa login`
- kalimat kontras, seperti `awalnya bagus tapi sekarang sering error`
- slang dan singkatan, seperti `bgt`, `gk`, `ga`, `nggak`, `gabisa`
- review sarkastik atau ambigu

Contoh kasus:

```text
Review: "Aplikasinya keren, tapi setelah update malah tidak bisa login."
Prediksi model: positif
Label yang lebih tepat: negatif
Kemungkinan penyebab: kata "keren" terlalu dominan, sedangkan konteks kontras setelah kata "tapi" belum dipahami dengan baik.
```

Hasil error analysis bisa dipakai untuk memperbaiki preprocessing, memperkaya lexicon, atau memilih model yang lebih peka terhadap konteks.

### 2. Data Augmentation dan Preprocessing Lanjutan

Dataset bisa diperkaya tanpa scraping ulang dengan data augmentation.

Teknik yang relevan:

- synonym replacement
- random insertion
- random deletion
- back-translation
- paraphrasing

Contoh:

```text
Teks asli: "makasih bgt ya!"
Hasil normalisasi: "terima kasih banget ya!"
Hasil back-translation: "terima kasih banyak ya!"
```

Preprocessing lanjutan:

- normalisasi slang Indonesia
- stemming atau lemmatization
- emoji handling
- deteksi negasi
- penanganan kata kontras seperti `tapi`, `namun`, dan `meskipun`

### 3. Optimasi Hyperparameter

Eksperimen berikutnya bisa memakai tuning hyperparameter yang lebih sistematis.

Metode yang relevan:

- GridSearchCV
- RandomizedSearchCV
- Optuna
- Bayesian Optimization

Parameter yang bisa dioptimasi:

- `ngram_range`
- `max_features`
- jumlah komponen `TruncatedSVD`
- ukuran hidden layer MLP
- dropout
- learning rate
- batch size
- jumlah epoch

Contoh dokumentasi hasil tuning:

```text
Dropout 0.10 menghasilkan akurasi testing lebih stabil dibanding 0.30.
Dropout 0.30 terlalu membatasi model sehingga beberapa fitur penting tidak dipelajari optimal.
```

### 4. Interpretabilitas dan Visualisasi

Untuk memahami alasan model membuat prediksi tertentu, proyek bisa ditambah metode interpretabilitas.

Metode yang relevan:

- LIME
- SHAP
- analisis bobot fitur TF-IDF
- word cloud per kelas sentimen
- visualisasi PCA atau t-SNE

Contoh insight yang diharapkan:

```text
Kata "error", "gagal", dan "tidak bisa" memiliki pengaruh besar terhadap prediksi negatif.
Kata "mudah", "membantu", dan "lancar" berpengaruh besar terhadap prediksi positif.
```

Visualisasi ini membuat hasil model lebih mudah dipahami, tidak hanya berhenti di angka akurasi.

### 5. Manajemen Eksperimen dan Reproducibility

Eksperimen bisa dicatat dengan tools khusus agar hasil lebih mudah dibandingkan dan diulang.

Tools yang relevan:

- MLflow
- Weights & Biases

Informasi yang dicatat:

- konfigurasi model
- hyperparameter
- metrik training dan testing
- artifact model
- versi dataset
- confusion matrix

Dengan experiment tracking, alasan satu skema lebih baik dari skema lain bisa dijelaskan dengan bukti yang lebih rapi.

### 6. Automated Hyperparameter Optimization

Selain tuning manual, eksperimen bisa memakai HPO otomatis.

Tools yang relevan:

- Optuna
- KerasTuner

Contoh objective yang bisa dioptimasi:

```text
Mencari kombinasi learning rate, dropout, hidden layer, dan jumlah komponen SVD yang menghasilkan macro F1-score terbaik.
```

HPO membuat proses eksperimen lebih sistematis dan lebih dekat dengan workflow profesional.

### 7. Model yang Lebih Kontekstual

MLP dengan TF-IDF ringan dan cepat, tetapi belum sepenuhnya memahami konteks kalimat.

Model yang bisa dicoba:

- IndoBERT
- IndoBERTweet
- multilingual BERT
- IndoBERT fine-tuning untuk klasifikasi sentimen

Model transformer biasanya lebih kuat untuk kasus:

- negasi
- sarkasme
- kalimat panjang
- konteks kontras
- variasi bahasa informal

### 8. Deployment dan Serving Model

Agar model bisa digunakan orang lain, pipeline ini bisa dikembangkan menjadi aplikasi atau API.

Pilihan deployment:

- FastAPI untuk REST API
- Streamlit untuk demo web sederhana
- Docker untuk packaging environment
- TorchServe untuk serving model PyTorch

Contoh endpoint:

```text
POST /predict
Input : "Aplikasi sering error saat login"
Output: {"sentiment": "negatif"}
```

## Lisensi

Proyek ini dibuat untuk pembelajaran dan portofolio. Feel free untuk di-clone, di-fork, atau dijadikan referensi belajar.
