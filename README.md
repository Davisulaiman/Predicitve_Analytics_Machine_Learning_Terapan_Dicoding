# Prediksi Risiko Kanker Paru Berdasarkan Faktor Kesehatan

## 1. Domain Proyek

Kanker paru-paru merupakan salah satu penyebab utama kematian akibat kanker di seluruh dunia. Deteksi dini terhadap potensi risiko kanker paru dapat meningkatkan kemungkinan penyembuhan melalui penanganan yang tepat. Oleh karena itu, proyek ini bertujuan mengembangkan model machine learning untuk memprediksi risiko kanker paru berdasarkan sejumlah indikator seperti kebiasaan merokok, batuk kronis, usia, dan gaya hidup.

### Referensi:

* World Health Organization (WHO) - Lung Cancer Fact Sheet
* [Kaggle: Lung Cancer Prediction Dataset](https://www.kaggle.com/datasets) *(sumber asli jika tersedia)*

## 2. Business Understanding

### Problem Statement:

Bagaimana memanfaatkan data faktor risiko individu untuk memprediksi kemungkinan seseorang mengidap kanker paru-paru secara akurat?

### Goals:

* Membangun model prediksi untuk mengklasifikasikan apakah seseorang berisiko kanker paru atau tidak.
* Menentukan model terbaik berdasarkan metrik akurasi dan evaluasi lainnya.

### Solution Statement:

Kami mengembangkan dan membandingkan beberapa algoritma Machine Learning, yaitu:

1. Random Forest Classifier
2. Gradient Boosting Classifier
3. Neural Network (MLPClassifier)

Model terbaik dipilih berdasarkan nilai akurasi tertinggi. Selain itu, dilakukan juga tuning parameter dan evaluasi visual.

## 3. Data Understanding

### Dataset Summary:

* Jumlah data: 1000+ baris
* Target: `LUNG_CANCER` (Yes/No)
* Fitur:

  * `GENDER`, `AGE`, `SMOKING`, `YELLOW_FINGERS`, `ANXIETY`, `PEER_PRESSURE`, `CHRONIC_DISEASE`, `FATIGUE`, `ALLERGY`, `WHEEZING`, `ALCOHOL_CONSUMING`, `COUGHING`, `SHORTNESS_OF_BREATH`, `SWALLOWING_DIFFICULTY`, `CHEST_PAIN`

### Visualisasi:

EDA dilakukan dengan seaborn untuk memahami korelasi fitur, distribusi umur, serta pengaruh fitur-fitur terhadap target.

## 4. Data Preparation

* Encode fitur kategorikal menggunakan LabelEncoder.
* Scaling data menggunakan MinMaxScaler.
* Membagi data menjadi training dan testing (80:20).

**Alasan:** Encoding diperlukan agar model memahami input numerik, dan scaling digunakan untuk menyamakan skala data agar model convergen dengan lebih stabil.

## 5. Modeling

### Model yang digunakan:

* **Random Forest**

  * n\_estimators=100
  * max\_depth=None
* **Gradient Boosting**

  * learning\_rate=0.1
  * n\_estimators=100
* **MLPClassifier (Neural Network)**

  * hidden\_layer\_sizes=(100,)
  * activation='relu'

### Perbandingan Model:

| Model             | Akurasi (%) |
| ----------------- | ----------- |
| Random Forest     | 97.5%       |
| Gradient Boosting | 95.0%       |
| Neural Network    | 90.0%       |

Model terbaik: **Random Forest** karena memiliki akurasi tertinggi serta kestabilan prediksi yang lebih baik di confusion matrix.

## 6. Evaluation

### Metrik Evaluasi:

* Accuracy
* Confusion Matrix
* Classification Report (Precision, Recall, F1-score)

### Penjelasan Metrik:

* **Accuracy** = (TP + TN) / (Total)
* Cocok digunakan karena dataset relatif seimbang (antara penderita dan non-penderita)

## 7. Visualisasi Perbandingan Akurasi

![Model Accuracy Comparison](./accuracy_comparison.png)

> Grafik menunjukkan bahwa Random Forest mengungguli model lain dalam hal akurasi.

---

## Resources

* Dataset: Sudah disertakan (`dataset.csv`)
* Notebook: `Predictive_Analysis.ipynb`
* Libraries: `sklearn`, `pandas`, `matplotlib`, `xgboost`, `seaborn`, `imbalanced-learn`

---

Silakan jalankan `Predictive_Analysis.ipynb` untuk melihat seluruh pipeline dari preprocessing hingga evaluasi model.
