# Laporan Proyek Machine Learning – Prediksi Risiko Kanker Paru-Paru

## Domain Proyek

Kanker paru-paru merupakan salah satu jenis kanker paling mematikan di dunia, dengan lebih dari 2,2 juta kasus baru setiap tahun dan tingkat kelangsungan hidup lima tahun yang hanya sekitar 25%. Tingkat kematian yang tinggi ini disebabkan oleh keterlambatan diagnosis dan kerumitan karakteristik sel kanker, seperti heterogenitas intra-tumor dan resistensi obat. Oleh karena itu, diperlukan pendekatan baru yang lebih canggih dan efisien untuk mendeteksi dan memprediksi risiko kanker paru-paru sejak dini.

Machine Learning (ML) menjadi salah satu pendekatan paling menjanjikan untuk menjawab tantangan tersebut. ML mampu menganalisis berbagai jenis data baik klinis, imaging (CT, histopatologi), maupun omics (seperti RNA-seq, cfDNA, dan DNA methylation) untuk mendeteksi pola yang tidak mudah dikenali oleh manusia. Model ML tidak hanya membantu deteksi dini, tetapi juga digunakan dalam klasifikasi subtipe kanker (misalnya LUAD dan LUSC), prediksi respon terapi, hingga penentuan prognosis pasien secara presisi.

Gao et al. (2023) menunjukkan bagaimana ML dapat memprediksi keberhasilan imunoterapi kanker paru melalui pemodelan biomarker penting seperti PD-L1, Tumor Mutation Burden (TMB), dan karakteristik mikro-lingkungan tumor (TME). Dritsas & Trigka (2022) berhasil mengembangkan model berbasis data klinis sederhana untuk mendeteksi kanker paru dengan akurasi tinggi menggunakan Rotation Forest. Sementara itu, Li et al. (2022) menyoroti pentingnya integrasi data besar dari imaging dan -omics untuk memperkuat diagnosis, klasifikasi, dan personalisasi terapi kanker paru-paru.

Proyek ini berangkat dari kebutuhan tersebut: membangun model klasifikasi risiko kanker paru berbasis data klinis non-invasif yang ringan namun informatif. Dengan pendekatan ini, hasil prediksi dapat dimanfaatkan sebagai sistem skrining awal yang hemat biaya dan mudah diimplementasikan dalam sistem kesehatan primer.

**Referensi Ilmiah:**

* Gao et al., 2023. *Artificial Intelligence and Machine Learning in Lung Cancer Immunotherapy*. Journal of Hematology & Oncology, 16(55). [https://doi.org/10.1186/s13045-023-01456-y](https://doi.org/10.1186/s13045-023-01456-y)
* Dritsas & Trigka, 2022. *Lung Cancer Risk Prediction with Machine Learning Models*. BDCC, 6(139). [https://doi.org/10.3390/bdcc6040139](https://doi.org/10.3390/bdcc6040139)
* Li et al., 2022. *Machine Learning for Lung Cancer Diagnosis, Treatment, and Prognosis*. Genomics, Proteomics & Bioinformatics, 20(5), 850–866. [https://doi.org/10.1016/j.gpb.2022.11.003](https://doi.org/10.1016/j.gpb.2022.11.003)

---

## Business Understanding

### Problem Statements

* Bagaimana mengidentifikasi individu dengan risiko tinggi terkena kanker paru-paru berdasarkan data klinis sederhana?
* Algoritma klasifikasi ML apa yang memberikan akurasi terbaik dalam prediksi kanker paru-paru?

### Goals

* Membangun model klasifikasi risiko kanker paru berdasarkan data non-invasif.
* Membandingkan performa beberapa algoritma klasifikasi ML dan memilih model terbaik.

### Solution Statements

* Menggunakan 7 algoritma: Logistic Regression, Random Forest, XGBoost, KNN, Decision Tree, Gradient Boosting, dan AdaBoost.
* Evaluasi menggunakan 4 metrik utama: Accuracy, Precision, Recall, dan F1-score.
* Pemilihan model terbaik berdasarkan kombinasi metrik tertinggi.

---

## Data Understanding

Dataset yang digunakan berasal dari [Kaggle - Lung Cancer Dataset](https://www.kaggle.com/datasets/akashnath29/lung-cancer-dataset), yang berisi data klinis non-imaging dengan total:

* **Jumlah entri**: 3000 observasi
* **Jumlah fitur**: 15 fitur input dan 1 target (`LUNG_CANCER`)
* **Jenis data**: Data kategori dan numerik, mencakup usia, jenis kelamin, riwayat merokok, gejala seperti batuk, sesak napas, nyeri dada, serta kebiasaan hidup lainnya.

### 1. Struktur Data

Setelah membaca dataset menggunakan `pd.read_csv()`, informasi umum dataset adalah sebagai berikut:

* **Jumlah baris dan kolom**: 3000 baris dan 16 kolom
* **Nama-nama fitur**:

  * `AGE`, `GENDER`, `SMOKING`, `YELLOW_FINGERS`, `ANXIETY`, `PEER_PRESSURE`, `CHRONIC DISEASE`, `FATIGUE`, `ALLERGY`, `WHEEZING`, `ALCOHOL CONSUMING`, `COUGHING`, `SHORTNESS OF BREATH`, `SWALLOWING DIFFICULTY`, `CHEST PAIN`, dan `LUNG_CANCER` sebagai target variabel.

### 2. Duplikasi

```python
print("Jumlah duplikasi: ", df.duplicated().sum())
```

Terdapat **n** data duplikat yang telah dihapus dengan:

```python
df = df.drop_duplicates()
```

### 3. Missing Values

```python
df.isnull().sum()
```

Hasilnya menunjukkan bahwa **tidak ada nilai yang hilang** (missing values) di seluruh kolom dataset.

### 4. Statistik Deskriptif

```python
df.describe()
```

Statistik deskriptif dilakukan untuk melihat persebaran nilai numerik seperti usia (`AGE`). Nilai `AGE` berada dalam rentang yang wajar dan tidak terdapat anomali ekstrim.

### 5. Outlier

Meskipun tidak ditampilkan dalam notebook, analisis outlier dapat dilakukan menggunakan boxplot terhadap kolom numerik `AGE`. Namun berdasarkan ringkasan statistik (`describe()`), nilai-nilai tersebut masih dalam batas normal.

### 6. Karakteristik Fitur

Setiap fitur akan dijelaskan sebagai berikut:

* **`AGE`**: Numerik, usia pasien.
* **`GENDER`**: Kategori, terdiri dari nilai 'MALE' dan 'FEMALE'.
* **`SMOKING` hingga `CHEST PAIN`**: Fitur biner (1/0) yang menunjukkan keberadaan kondisi atau kebiasaan.
* **`LUNG_CANCER`**: Target variabel biner yang mengindikasikan apakah pasien terdiagnosis kanker paru-paru atau tidak.

### 7. Status Penggunaan Fitur

* **Fitur digunakan**: Semua fitur awal akan dianalisis dalam tahap eksplorasi awal.
* **Fitur tidak digunakan**: Jika dalam proses analisis selanjutnya terdapat fitur dengan korelasi rendah atau multikolinearitas tinggi, fitur tersebut dapat dieliminasi.
* **Fitur yang akan dihapus**: Belum ada yang dihapus pada tahap ini, namun pencatatan duplikasi sudah dilakukan.

---

### Variabel:

* Numerik: `AGE`
* Kategorikal/Biner: `GENDER`, `SMOKING`, `YELLOW_FINGERS`, `ANXIETY`, `COUGHING`, dll.
* Target: `LUNG_CANCER` (Yes/No)

### Visualisasi Distribusi Target

![Distribusi Target](assets/target_distribution.png)

*Gambar 1. Distribusi kelas target: jumlah kasus kanker paru-paru positif dan negatif.*

### Heatmap Korelasi

![Heatmap Korelasi](assets/heatmap_correlation.png)

*Gambar 2. Korelasi antar fitur prediktor terhadap variabel target.*


---

## Data Preparation

Proses *Data Preparation* dilakukan dalam beberapa tahapan yang disusun sesuai dengan alur kerja di dalam notebook. Setiap langkah memiliki tujuan tertentu yang mendukung performa model machine learning serta memastikan kualitas data yang optimal.

### 1. Data Cleaning

Langkah pertama dalam proses persiapan data adalah **menghapus data duplikat** untuk mencegah bias model akibat observasi yang berulang.

```python
df = df.drop_duplicates()
```

Jumlah data duplikat yang terdeteksi sebelumnya adalah **n entri**, dan telah dihapus.

### 2. Encoding Fitur Kategorikal

Seluruh fitur kategorikal dikonversi menggunakan **Label Encoding**, karena sebagian besar fitur biner (bernilai 0 dan 1) atau bertipe kategori sederhana.

```python
le = LabelEncoder()
for col in categorical_columns:
    df[col] = le.fit_transform(df[col])
```

### 3. Feature Scaling (Standarisasi)

Fitur numerik seperti `AGE` mengalami proses **standarisasi** menggunakan `StandardScaler`, karena model berbasis jarak seperti K-Nearest Neighbors (KNN) sensitif terhadap skala fitur.

```python
scaler = StandardScaler()
df['AGE'] = scaler.fit_transform(df[['AGE']])
```

> **Catatan:**
>
> * **Standarisasi** = mengubah distribusi data agar memiliki **rata-rata 0** dan **standar deviasi 1**.
> * **Normalisasi** (misalnya `MinMaxScaler`) digunakan bila ingin memetakan nilai ke rentang tertentu, seperti 0–1.
>   Pada tahap ini, yang dilakukan adalah **standarisasi**, bukan normalisasi.

### 4. Train-Test Split

Setelah data bersih dan siap digunakan, dataset dibagi menjadi dua bagian:

* **80% untuk pelatihan (training)**
* **20% untuk pengujian (testing)**

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

### Ringkasan Alur Data Preparation:

| No | Langkah          | Deskripsi                                                                |
| -- | ---------------- | ------------------------------------------------------------------------ |
| 1  | Data Cleaning    | Penghapusan duplikasi data                                               |
| 2  | Encoding         | Konversi fitur kategorikal menggunakan LabelEncoder                      |
| 3  | Standarisasi     | Skala numerik distandarkan untuk mendukung performa model berbasis jarak |
| 4  | Train-Test Split | Pembagian data untuk evaluasi performa model                             |


---

## Modeling

Tahap *Modeling* merupakan inti dari proses prediksi, di mana berbagai algoritma pembelajaran mesin digunakan untuk mempelajari pola dari data pelatihan dan melakukan prediksi terhadap data pengujian. Berikut adalah model-model yang digunakan, beserta cara kerja, parameter penting, dan alasan pemilihannya.

### 1. Logistic Regression

**Deskripsi:**
Model klasifikasi biner yang memodelkan probabilitas kejadian berdasarkan fungsi logistik. Logistic Regression sangat efisien untuk dataset kecil hingga menengah dan bekerja baik jika hubungan antar fitur dan target bersifat linear.

**Parameter:**

* `penalty`: regularisasi L2 (default)
* `C=1.0`: parameter regularisasi (semakin kecil, regularisasi semakin kuat)
* `solver='lbfgs'`
* `max_iter=10000`: digunakan agar model cukup waktu untuk konvergen

**Alasan pemilihan:**
Digunakan sebagai baseline model karena efisien, cepat, dan mudah diinterpretasikan.

---

### 2. K-Nearest Neighbors (KNN)

**Deskripsi:**
Model berbasis instance yang mengklasifikasi data baru berdasarkan mayoritas kelas dari k tetangga terdekat dalam ruang fitur. Tidak membutuhkan pelatihan eksplisit.

**Parameter:**

* `n_neighbors=10`: jumlah tetangga terdekat
* `metric='minkowski'` (default), menggunakan Euclidean distance

**Alasan pemilihan:**
Mudah diimplementasikan dan cocok untuk memahami performa awal model berbasis jarak.

---

### 3. Decision Tree

**Deskripsi:**
Model yang memecah data secara rekursif berdasarkan fitur yang paling mengurangi impuritas. Cocok untuk data non-linear dan mudah diinterpretasikan.

**Parameter:**

* `criterion='gini'` (default): pengukuran impuritas
* `max_depth`: tidak diatur, memungkinkan pohon tumbuh penuh (berpotensi overfitting)

**Alasan pemilihan:**
Interpretabilitas tinggi dan cepat dalam inferensi.

---

### 4. Random Forest

**Deskripsi:**
Model ansambel berbasis banyak pohon keputusan (decision trees) yang dilatih pada subset data dan fitur yang berbeda, lalu hasilnya dirata-rata. Mengurangi overfitting dibandingkan decision tree tunggal.

**Parameter:**

* `n_estimators=100`: jumlah pohon dalam hutan
* `max_features='auto'`: fitur dipilih secara acak

**Alasan pemilihan:**
Lebih stabil, tahan terhadap overfitting, dan cocok untuk dataset menengah.

---

### 5. Gradient Boosting

**Deskripsi:**
Model ansambel yang membangun model secara bertahap, memperbaiki kesalahan prediksi model sebelumnya menggunakan pendekatan *gradient descent*.

**Parameter:**

* `n_estimators=100`, `learning_rate=0.1`

**Alasan pemilihan:**
Sangat akurat dalam banyak kompetisi klasifikasi, walaupun lebih lambat dibanding Random Forest.

---

### 6. AdaBoost

**Deskripsi:**
Model boosting yang melatih model sederhana secara berurutan dan menyesuaikan bobot pada data yang sebelumnya salah prediksi.

**Parameter:**

* `n_estimators=50`, `learning_rate=1.0`

**Alasan pemilihan:**
Mengatasi bias, terutama ketika model dasar lemah, dan cukup efisien.

---

### 7. XGBoost

**Deskripsi:**
Varian efisien dari gradient boosting dengan kemampuan regularisasi tambahan dan optimasi yang lebih baik. Sangat populer dalam kompetisi ML karena akurasi tinggi.

**Parameter:**

* `use_label_encoder=False`
* `eval_metric='logloss'`
* `n_estimators=100`, `learning_rate=0.1`

**Alasan pemilihan:**
Presisi tinggi dan efisien dalam menangani missing values serta fitur dalam skala besar.

---

### Ringkasan Perbandingan Model

| Algoritma           | Kelebihan            | Kekurangan                                   |
| ------------------- | -------------------- | -------------------------------------------- |
| Logistic Regression | Cepat, sederhana     | Tidak cocok untuk relasi non-linear kompleks |
| KNN                 | Mudah dipahami       | Tidak efisien di dataset besar               |
| Decision Tree       | Mudah diinterpretasi | Rentan overfitting                           |
| Random Forest       | Stabil dan akurat    | Sulit diinterpretasi                         |
| Gradient Boosting   | Sangat akurat        | Kompleks dan lambat                          |
| AdaBoost            | Mengatasi bias       | Sensitif terhadap outlier                    |
| XGBoost             | Sangat presisi       | Perlu tuning parameter                       |

---
## Evaluation

### Metrik Evaluasi yang Digunakan:

Untuk mengevaluasi performa model klasifikasi, digunakan empat metrik utama:

* **Accuracy** = (TP + TN) / (Total)
* **Precision** = TP / (TP + FP)
* **Recall** = TP / (TP + FN)
* **F1-score** = 2 × (Precision × Recall) / (Precision + Recall)

Metrik ini dipilih karena:

* **Precision** berguna untuk mengetahui seberapa tepat model dalam memprediksi kanker (menghindari false positive).
* **Recall** sangat penting dalam konteks medis karena menunjukkan seberapa baik model mendeteksi pasien yang benar-benar menderita kanker (menghindari false negative).
* **F1-score** menyatukan kedua aspek tersebut dalam satu nilai harmonis.
* **Accuracy** menunjukkan performa keseluruhan namun bisa menyesatkan jika data tidak seimbang — oleh karena itu, digunakan bersama metrik lainnya.

---

### Visualisasi Perbandingan Metrik Model

![Perbandingan Akurasi](assets/accuracy_chart.png)

*Gambar 3. Visualisasi perbandingan skor akurasi antar model ML.*

### Confusion Matrix

![Confusion Matrix ](assets/confusion_matrix.png)

*Gambar 4. Confusion matrix pada model – performa klasifikasi.*

---


### Hasil Evaluasi Model

| Model               | Accuracy   | Precision  | Recall     | F1-Score   |
| ------------------- | ---------- | ---------- | ---------- | ---------- |
| Logistic Regression | 0.9700     | 0.9700     | 0.9700     | 0.9700     |
| Random Forest       | **0.9817** | **0.9818** | **0.9817** | **0.9817** |
| XGBoost             | 0.9783     | 0.9784     | 0.9783     | 0.9783     |
| KNN                 | 0.9750     | 0.9750     | 0.9750     | 0.9750     |
| Decision Tree       | 0.9683     | 0.9688     | 0.9683     | 0.9684     |
| Gradient Boosting   | 0.9800     | 0.9802     | 0.9800     | 0.9800     |
| AdaBoost            | 0.9750     | 0.9750     | 0.9750     | 0.9750     |

---



### Model Terbaik: **Random Forest**

Model Random Forest dipilih sebagai **model terbaik** karena:

* Memiliki nilai **akurasi tertinggi (98.17%)**
* Menunjukkan keseimbangan tinggi antara precision dan recall
* Stabil terhadap data baru karena sifat ansambel dari banyak pohon keputusan
* Berperforma konsisten dalam deteksi kasus kanker paru-paru, yang sangat penting dalam konteks medis

---

## Keterkaitan dengan Business Understanding

### Apakah Model Menjawab Problem Statements?

✅ **Ya.**
Model yang dibangun mampu mengidentifikasi individu dengan risiko tinggi terkena kanker paru-paru **hanya berdasarkan data klinis non-invasif**, menjawab problem pertama secara langsung.

✅ **Ya.**
Dengan membandingkan performa tujuh algoritma ML, laporan ini berhasil mengidentifikasi **Random Forest sebagai model dengan performa terbaik**, menjawab problem kedua.

---

### Apakah Model Mencapai Goals?

✅ **Membangun Model Klasifikasi:**
Model dikembangkan menggunakan data klinis yang sederhana dan menghasilkan **akurasi tinggi** di atas 97% pada semua model utama.

✅ **Pemilihan Model Terbaik:**
Dilakukan evaluasi menyeluruh terhadap tujuh algoritma menggunakan metrik evaluasi utama. Pemilihan **Random Forest** didasarkan pada hasil evaluasi kuantitatif yang solid.

---

### Apakah Solusi yang Diberikan Berdampak?

✅ **Ya.**
Solusi yang dirancang berdampak signifikan karena:

* Memberikan dasar sistem **skrining dini** terhadap kanker paru-paru.
* Dapat diimplementasikan di fasilitas kesehatan dengan data yang mudah dikumpulkan.
* Berpotensi menurunkan angka keterlambatan diagnosis melalui **deteksi berbasis data klinis sederhana**.

---

## Kesimpulan

* **Random Forest** memberikan performa terbaik secara keseluruhan dalam mendeteksi risiko kanker paru-paru.
* Pendekatan ML terbukti mampu menyelesaikan tantangan prediksi medis secara efisien dan akurat.
* Evaluasi menunjukkan bahwa seluruh problem, goals, dan solusi yang dirumuskan di awal berhasil dijawab dan tercapai.

---

## Saran Pengembangan

* Gunakan **K-Fold Cross Validation** untuk menghindari bias dari split data tunggal.
* Eksplorasi fitur tambahan dari data **imaging** dan **genomik** untuk akurasi lebih tinggi.
* Pertimbangkan pengembangan menjadi **aplikasi prediktif** berbasis web/mobile untuk screening kanker paru-paru secara masif dan mudah diakses.
