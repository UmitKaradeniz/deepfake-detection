# Sentetik Medya Tespitinde İşitsel-Görsel Tutarlılık ve Davranışsal Biyometri Analizi

## 1. PROJE AMACI

Bu proje, Python tabanlı bir **Deepfake tespit sistemi** geliştirmeyi amaçlar. Sistem, bir videonun **dudak-ses senkronizasyonunu** analiz ederek "Gerçek" veya "Sahte" skoru üretir. **Uç cihazlarda (mobil/CPU)** çalışabilecek kadar hafif bir mimari hedeflenmiştir — ağır CNN (ResNet, EfficientNet) veya Transformer modelleri kullanılmamıştır.

---

## 2. MİMARİ GENEL BAKIŞ

```
┌──────────────────────────────────────────────────────────────┐
│                        VİDEO GİRDİSİ                         │
│                    (MP4 dosyası / kamera)                     │
└──────────────┬───────────────────────────┬───────────────────┘
               │                           │
     ┌─────────▼─────────┐      ┌─────────▼──────────┐
     │   GÖRSEL ANALİZ    │      │   İŞİTSEL ANALİZ    │
     │   MediaPipe Face   │      │   Librosa            │
     │   Mesh (468 lm)    │      │   FFmpeg             │
     │   ↓                │      │   ↓                  │
     │   24 Action Unit   │      │   13 MFCC +          │
     │   mesafesi          │      │   7 Spectral = 20    │
     └─────────┬──────────┘      └──────────┬──────────┘
               │    Timestamp Senkronizasyonu │
               │    (frame_idx × sr / fps)    │
     ┌─────────▼──────────────────────────────▼──────────┐
     │              HİBRİT MODEL (1D-CNN + LSTM)          │
     │  ┌─────────────┐              ┌─────────────┐      │
     │  │ Görsel Dal   │              │ İşitsel Dal  │      │
     │  │ Conv1D(64,3)│              │ Conv1D(64,3) │      │
     │  │ Conv1D(32,3)│              │ Conv1D(32,3) │      │
     │  │ LSTM(64)    │              │ LSTM(64)     │      │
     │  └──────┬──────┘              └──────┬──────┘      │
     │         └──────────┬─────────────────┘              │
     │              Concatenate (Füzyon)                    │
     │              Dense(128) → Dense(64) → Dense(32)     │
     │              Sigmoid → [0,1] Deepfake Skoru         │
     └─────────────────────────────────────────────────────┘
```

---

## 3. KULLANILAN TEKNOLOJİLER

| Teknoloji        | Versiyon | Kullanım Amacı                                            |
| ---------------- | -------- | --------------------------------------------------------- |
| Python           | 3.13.9   | Ana dil                                                   |
| TensorFlow/Keras | 2.x      | Model eğitimi (LSTM, Conv1D, Dense)                       |
| MediaPipe        | 0.10.32  | 468 yüz landmark çıkarma (Face Mesh)                      |
| Librosa          | -        | MFCC ve spektral ses analizi                              |
| OpenCV           | -        | Video kare okuma, renk dönüşümü                           |
| NumPy            | -        | Sayısal hesaplamalar, veri saklama                        |
| scikit-learn     | -        | StandardScaler, RandomForest, GradientBoosting (deneysel) |
| XGBoost          | -        | Gradient boosting (deneysel)                              |
| imageio-ffmpeg   | -        | Videodan ses çıkarma                                      |
| Matplotlib       | -        | Eğitim grafikleri                                         |

---

## 4. VERİ SETİ: FakeAVCeleb v1.2

- **Konum**: `C:\Users\dkara\Desktop\FakeAVCeleb_v1.2`
- **Toplam**: 21,566 video
- **Kategoriler**:
  - `RealVideo-RealAudio`: 500 video (Gerçek → label=0)
  - `RealVideo-FakeAudio`: 500 video (Sahte → label=1)
  - `FakeVideo-RealAudio`: 9,709 video (Sahte → label=1)
  - `FakeVideo-FakeAudio`: 10,857 video (Sahte → label=1)
- **Sınıf dengesizliği**: 500 real vs 20,566 fake → **Dengeli alt küme: 500 real + 500 fake**
- **Bölme**: %80 train (800) / %20 test (200)
- **Metadata**: `meta_data.csv` dosyasından okunuyor

---

## 5. PROJE DOSYA YAPISI

```
c:\Users\dkara\Desktop\New folder\
├── .venv/                          # Python sanal ortam
├── egitim_verisi/                  # İşlenmiş .npy dosyaları
│   ├── X_vis_train.npy             # Görsel özellikler (train)
│   ├── X_aud_train.npy             # İşitsel özellikler (train)
│   ├── y_train.npy                 # Etiketler (train)
│   ├── X_vis_test.npy              # Görsel özellikler (test)
│   ├── X_aud_test.npy              # İşitsel özellikler (test)
│   ├── y_test.npy                  # Etiketler (test)
│   ├── vis_mean.npy / vis_std.npy  # Görsel normalizasyon parametreleri
│   └── aud_mean.npy / aud_std.npy  # İşitsel normalizasyon parametreleri
├── dataset_loader.py               # FakeAVCeleb CSV okuma, dengeleme, train/test split
├── create_dataset.py               # Video → özellik çıkarma (AU + MFCC)
├── train_model.py                  # 1D-CNN + LSTM hibrit model eğitimi
├── predict_video.py                # Video üzerinde tahmin yapma
├── face_landmarker.task            # MediaPipe Face Mesh modeli (3.7MB)
├── deepfake_detector_model.h5      # Eğitilmiş model (HDF5)
├── deepfake_detector_best.keras    # En iyi epoch modeli
├── deepfake_detector_rf.pkl        # Random Forest/XGBoost modeli (deneysel)
├── training_results.png            # Accuracy/Loss grafikleri
├── feature_importance.png          # Feature importance grafiği
├── audio_analysis.py               # Eski ses analiz scripti (kullanılmıyor)
├── feature_extraction.py           # Eski özellik çıkarma (kullanılmıyor)
├── debug_mp.py                     # MediaPipe debug scripti
├── check_tasks_api.py              # MediaPipe Tasks API kontrol
├── verify_audio.py                 # Ses bağımlılık kontrolü
├── verify_install.py               # Kurulum doğrulama
└── download_model.py               # Face Mesh model indirme
```

**Aktif dosyalar (pipeline)**: `dataset_loader.py` → `create_dataset.py` → `train_model.py` → `predict_video.py`

---

## 6. DETAYLI DOSYA AÇIKLAMALARI

### 6.1. dataset_loader.py

FakeAVCeleb v1.2 veri setini yükler.

**Fonksiyonlar:**

- `load_metadata()`: `meta_data.csv`'yi okur, video yollarını oluşturur
- `filter_existing_videos()`: Dosyası olmayan videoları filtreler
- `create_balanced_subset()`: 500 real + 500 fake dengeli set oluşturur
- `train_test_split()`: Stratified %80/%20 bölme
- `get_dataset(samples_per_class)`: Ana fonksiyon, yukarıdakileri sırayla çağırır

**Etiketleme mantığı:**

- `RealVideo-RealAudio` → label=0 (Gerçek)
- Diğer tüm kombinasyonlar → label=1 (Sahte)

### 6.2. create_dataset.py

Videoları işleyerek özellik çıkarır.

**Görsel özellikler (24 Action Unit):**
MediaPipe Face Mesh'in 468 landmark noktasından hesaplanan FACS tabanlı mesafeler:

| Grup         | Özellikler                                                    | Sayı |
| ------------ | ------------------------------------------------------------- | ---- |
| Dudak AU     | Açıklık, genişlik, oran, dış kenar, iç köşeler, eğriler, çene | 8    |
| Göz AU       | Sol/sağ açıklık, EAR, genişlik, simetri, ort. EAR             | 8    |
| Kaş AU       | Sol/sağ kaş-göz mesafesi, iç kaş, simetri                     | 4    |
| Yüz Geometri | Burun-çene, alın-burun, yüz oranı, dudak-yüz oranı            | 4    |

**İşitsel özellikler (20):**

| Grup     | Özellikler                                                 | Sayı |
| -------- | ---------------------------------------------------------- | ---- |
| MFCC     | Mel-Frequency Cepstral Coefficients                        | 13   |
| Spectral | Centroid, Rolloff, Bandwidth, ZCR, RMS, Flatness, Contrast | 7    |

**Zaman senkronizasyonu:**

```python
audio_start = frame_idx * samples_per_frame
audio_end = audio_start + audio_chunk
# Her video karesine karşılık gelen ses segmenti alınır
```

**İşleme parametreleri:**

- `SEQUENCE_LENGTH = 30` (kayan pencere boyutu)
- `FRAME_SKIP = 2` (her 2. kare)
- `MAX_DURATION_SEC = 3` (video başına max süre)

**Çıktı**: 6 adet `.npy` dosyası (X_vis_train, X_aud_train, y_train, X_vis_test, X_aud_test, y_test)

### 6.3. train_model.py

1D-CNN + LSTM hibrit model eğitimi.

**Model mimarisi:**

```
Görsel Dal: Input(30,24) → Conv1D(64,3) → BN → Drop(0.3) → Conv1D(32,3) → BN → LSTM(64) → Drop(0.3)
İşitsel Dal: Input(30,20) → Conv1D(64,3) → BN → Drop(0.3) → Conv1D(32,3) → BN → LSTM(64) → Drop(0.3)
Füzyon: Concatenate → Dense(128) → Drop(0.4) → Dense(64) → Drop(0.3) → Dense(32) → Sigmoid
```

**Eğitim detayları:**

- Optimizer: Adam (lr=0.0005)
- Loss: Binary Crossentropy
- Batch size: 32
- Max epochs: 100
- Data augmentation: Gaussian gürültü (×2 veri)
- Normalizasyon: Mean/Std (Z-score)
- L2 regularization: 0.001
- Callbacks: EarlyStopping(patience=15), ModelCheckpoint, ReduceLROnPlateau

### 6.4. predict_video.py

Eğitilmiş modelle video üzerinde tahmin yapar.

**Kullanım:**

```bash
python predict_video.py video.mp4
```

**Akış:**

1. Video kareleri → MediaPipe Face Mesh → 24 AU vektörü
2. Karşılık gelen ses segmenti → MFCC + Spectral → 20 ses vektörü
3. Normalizasyon (eğitim parametreleriyle)
4. Sliding window → Model tahmin
5. Tüm pencerelerin ortalaması → Deepfake Skoru (0-1)

**Çıktı:** Skor > 0.5 → Sahte, Skor ≤ 0.5 → Gerçek

---

## 7. EĞİTİM SONUÇLARI VE DENEMELER

Birçok farklı model ve özellik seti denendi:

| #   | Model                       | Özellik                 | Accuracy  |
| --- | --------------------------- | ----------------------- | --------- |
| 1   | LSTM (2 katman)             | 2 görsel + 13 MFCC      | %70.4     |
| 2   | LSTM + BatchNorm            | 15 görsel + 13 MFCC     | %73.7     |
| 3   | Conv1D + LSTM               | 15 görsel + 13 MFCC     | %74.6     |
| 4   | Gradient Boosting           | 624 temporal istatistik | %75.3     |
| 5   | XGBoost + Feature Selection | 200 seçili özellik      | %75.6     |
| 6   | **1D-CNN + LSTM (final)**   | **24 AU + 20 ses**      | **%74.1** |

**Gözlemler:**

- Tüm modeller %74-76 arasında tavan yaptı
- En iyi epoch genellikle 1. epoch oldu (sonra overfitting)
- Train accuracy %99'a çıkarken val accuracy %74'te kaldı
- Geometrik özelliklerle (landmark mesafeleri) erişilebilecek **tavan ~%75**
- Final model olarak proje spesifikasyonuna uygun **1D-CNN + LSTM hibrit** seçildi

**Sınırlamalar:**

- CNN kullanmadığı için piksel seviyesinde artefaktları yakalayamıyor
- Eğitim verisi sadece FakeAVCeleb ünlülerine ait (farklı ortam videolarında düşük güven)
- 500+500 video, derin öğrenme için az sayıda

---

## 8. İŞLENMİŞ VERİ BOYUTLARI

```
X_vis_train: (6368, 30, 24) → 6368 pencere, her biri 30 kare, 24 AU özellik
X_aud_train: (6368, 30, 20) → 6368 pencere, her biri 30 kare, 20 ses özellik
y_train:     (6368,)        → 0=Gerçek, 1=Sahte
X_vis_test:  (1584, 30, 24)
X_aud_test:  (1584, 30, 20)
y_test:      (1584,)
Eğitim dengesi: Real=3217, Fake=3151
```

---

## 9. BAĞIMLILIKLAR

```
pip install opencv-python mediapipe numpy librosa matplotlib tensorflow imageio-ffmpeg scikit-learn xgboost
```

Ek olarak `face_landmarker.task` dosyası gerekli (MediaPipe Face Mesh modeli).

---

## 10. ÇALIŞTIRMA TALİMATLARI

```bash
# 1. Sanal ortam oluştur ve aktifleştir
python -m venv .venv
.\.venv\Scripts\activate

# 2. Bağımlılıkları kur
pip install opencv-python mediapipe numpy librosa matplotlib tensorflow imageio-ffmpeg

# 3. Veri setini işle (FakeAVCeleb gerekli)
python create_dataset.py

# 4. Modeli eğit
python train_model.py

# 5. Video analiz et
python predict_video.py video.mp4
```

---

## 11. YAPILMASI GEREKENLER / GELİŞTİRME ALANLARI

- [ ] **Accuracy artırma**: CNN tabanlı özellik çıkarma (MobileNetV2)
- [ ] **TFLite dönüşümü**: Uç cihaz deployment için model conversion
- [ ] **Web arayüzü**: Flask/FastAPI ile video yükleme ve analiz
- [ ] **Raspberry Pi**: Gerçek zamanlı kamera analizi
- [ ] **Daha fazla veri**: 500+500 yerine tüm 21K video ile eğitim
- [ ] **3 kademeli sınıflandırma**: Gerçek / Belirsiz / Sahte güven aralıkları
- [ ] **Frekans domain analizi**: FFT/DCT ile deepfake artefakt tespiti
- [ ] **Optical flow**: Hareket tutarsızlığı analizi

---

## 12. ÖNEMLİ NOTLAR

1. **MediaPipe API**: `mp.solutions.face_mesh` DEPRECATED. Projede `mp.tasks.vision.FaceLandmarker` (Tasks API) kullanılıyor.
2. **Ses çıkarma**: `librosa.load()` direkt video açamıyor, `imageio-ffmpeg` ile WAV'a çeviriliyor.
3. **Model kaydetme**: `.h5` formatı legacy, `.keras` formatı öneriliyor (TensorFlow uyarısı).
4. **Python 3.13**: Bazı eski kütüphanelerle uyumsuzluk olabilir.
5. **face_landmarker.task**: MediaPipe'ın 468 noktalı face mesh modeli, proje klasöründe bulunmalı.
