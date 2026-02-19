# Sentetik Medya Tespitinde İşitsel-Görsel Tutarlılık ve Davranışsal Biyometri Analizi

## Mimari Açıklama Dokümanı

---

## 1. Proje Tanımı

Bu proje, bir videonun **deepfake** olup olmadığını tespit eden hafif (edge-device uyumlu) bir yapay zeka sistemidir. Ağır CNN (ResNet, EfficientNet) veya Transformer modelleri **kullanılmamıştır**. Bunun yerine MediaPipe Face Mesh'in geometrik çıktıları ve ses sinyalinin frekans analizi birlikte değerlendirilir.

**Temel Hipotez:** Gerçek bir videoda konuşmacının dudak hareketleri ile ses sinyali arasında doğal bir senkronizasyon vardır. Deepfake videolarda bu senkronizasyon bozulur. Ayrıca GAN tabanlı yüz sentezleme algoritmaları, frekans düzleminde insan gözünün göremediği artefaktlar bırakır.

---

## 2. Sistem Mimarisi (End-to-End Pipeline)

```
                         ┌───────────────────┐
                         │    VİDEO GİRDİSİ   │
                         │    (MP4 dosyası)    │
                         └────────┬──────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
          ┌─────────▼─────────┐       ┌─────────▼─────────┐
          │   GÖRSEL PIPELINE  │       │  İŞİTSEL PIPELINE  │
          │                   │       │                    │
          │  OpenCV:          │       │  FFmpeg:           │
          │  Video → Kareler  │       │  Video → WAV       │
          │                   │       │                    │
          │  MediaPipe:       │       │  Librosa:          │
          │  468 Landmark     │       │  WAV → Sinyal      │
          │       │           │       │       │            │
          │       ▼           │       │       ▼            │
          │  24 Action Unit   │       │  13 MFCC           │
          │  mesafesi (FACS)  │       │  7 Spectral feat.  │
          │       │           │       │  1 RMS enerji      │
          │       ▼           │       │                    │
          │  10 DCT frekans   │       │                    │
          │  özelliği         │       │                    │
          └────────┬──────────┘       └─────────┬──────────┘
                   │                            │
                   │   Timestamp Senkronizasyonu │
                   │   audio_idx = frame × (sr/fps)
                   │                            │
          ┌────────▼────────────────────────────▼──────────┐
          │            ÖZELLİK MÜHENDİSLİĞİ                │
          │                                                │
          │  Velocity (1. türev):   34 özellik             │
          │  Acceleration (2. türev): 34 özellik           │
          │  Sync Features:          3 özellik             │
          │  (Dudak×RMS korelasyon, uyumsuzluk, genişlik)  │
          │                                                │
          │  TOPLAM GÖRSEL: 34 + 34 + 34 + 3 = 105        │
          │  TOPLAM İŞİTSEL: 20                            │
          └────────┬────────────────────────────┬──────────┘
                   │                            │
          ┌────────▼──────────┐       ┌─────────▼──────────┐
          │   GÖRSEL DAL       │       │   İŞİTSEL DAL      │
          │                   │       │                    │
          │  Conv1D(64, k=3)  │       │  Conv1D(64, k=3)   │
          │  SE-Block(r=4)    │       │  SE-Block(r=4)     │
          │  BatchNorm        │       │  BatchNorm         │
          │  Dropout(0.3)     │       │  Dropout(0.3)      │
          │  Conv1D(32, k=3)  │       │  Conv1D(32, k=3)   │
          │  BatchNorm        │       │  BatchNorm         │
          │  LSTM(64, seq)    │       │  LSTM(64, seq)     │
          │  Dropout(0.3)     │       │  Dropout(0.3)      │
          │  Self-Attention   │       │  Self-Attention    │
          │       │           │       │       │            │
          └───────┤           │       └───────┤            │
                  └───────────┤───────────────┘
                              │
                    ┌─────────▼──────────┐
                    │  FÜZYON (Concat)    │
                    │                    │
                    │  Dense(128, ReLU)   │
                    │  Dropout(0.4)       │
                    │  Dense(64, ReLU)    │
                    │  Dropout(0.3)       │
                    │  Dense(32, ReLU)    │
                    │  Dense(1, Sigmoid)  │
                    │       │            │
                    │       ▼            │
                    │  Deepfake Skoru    │
                    │  [0.0 → 1.0]      │
                    │                    │
                    │  <0.35 = GERÇEK    │
                    │  0.35-0.65 = ?     │
                    │  >0.65 = SAHTE     │
                    └────────────────────┘
```

---

## 3. Veri Seti: FakeAVCeleb v1.2

### 3.1 Genel Bilgi

FakeAVCeleb, hem ses hem video manipülasyonu içeren ilk büyük ölçekli deepfake veri setidir.

| Kategori            | Video Sayısı | Label      | Açıklama                        |
| ------------------- | ------------ | ---------- | ------------------------------- |
| RealVideo-RealAudio | ~500         | 0 (Gerçek) | Orijinal ses + orijinal görüntü |
| RealVideo-FakeAudio | ~500         | 1 (Sahte)  | Orijinal görüntü + sentetik ses |
| FakeVideo-RealAudio | ~9,709       | 1 (Sahte)  | Sentetik görüntü + orijinal ses |
| FakeVideo-FakeAudio | ~10,857      | 1 (Sahte)  | Sentetik görüntü + sentetik ses |
| **Toplam**          | **~21,566**  |            |                                 |

### 3.2 Veri Dengeleme Stratejisi

Sınıf dengesizliği (500 real vs 21K fake) modelin her şeye "fake" demesine yol açar. Bu nedenle **dengeli alt küme** kullanılır: **500 Real + 500 Fake**, stratified %80/%20 train/test split.

### 3.3 Sliding Window

Her video, sabit uzunluklu zaman pencerelerine bölünür:

- `SEQUENCE_LENGTH = 30` kare
- `FRAME_SKIP = 2` (her 2. kare işlenir)
- Bir video → birden fazla eğitim örneği (örn. 185 kare → 155 pencere)
- Sonuç: 500+500 video → **~6,400 train + ~1,600 test penceresi**

---

## 4. Özellik Mühendisliği (Feature Engineering)

### 4.1 Görsel Özellikler — Action Units (24 özellik)

Google MediaPipe Face Mesh, bir yüzden **468 adet 3D landmark** noktası çıkarır. Bu noktalar arasındaki **Öklid mesafeleri** hesaplanarak Facial Action Coding System (FACS) benzetmesi yapılır.

#### Dudak Action Units (8 özellik)

```
Landmark 13 ─── Üst dudak merkez
    │
    │ Dikey açıklık (AU25-AU26: Lips Part / Jaw Drop)
    │
Landmark 14 ─── Alt dudak merkez

Landmark 78 ──────────── Landmark 308
    Sol köşe              Sağ köşe
         Yatay genişlik (AU12: Lip Corner Puller)
```

| #   | Özellik              | Formül                      | FACS Karşılığı |
| --- | -------------------- | --------------------------- | -------------- |
| 1   | Dudak dikey açıklık  | dist(13, 14)                | AU25+AU26      |
| 2   | Dudak yatay genişlik | dist(78, 308)               | AU12           |
| 3   | Dudak oran           | dikey / yatay               | Açıklık oranı  |
| 4   | Alt dudak-çene       | dist(0, 17)                 | AU27           |
| 5   | Dış dudak genişliği  | dist(61, 291)               | AU20           |
| 6-7 | İç dudak köşeleri    | dist(82,312), dist(87,317)  | AU28           |
| 8   | Çene-burun oranı     | dist(152,6) / yüz_genişliği | AU29           |

#### Göz Action Units (8 özellik)

```
    Landmark 159 (üst kapak)
         │
         │  Göz Açıklığı (EAR)
         │                     EAR = göz_yükseklik / göz_genişlik
    Landmark 145 (alt kapak)

Landmark 33 ───────── Landmark 133
  İç köşe              Dış köşe
```

| #     | Özellik                | FACS Karşılığı         |
| ----- | ---------------------- | ---------------------- |
| 9-10  | Sol/sağ göz yüksekliği | AU5 (Upper Lid Raiser) |
| 11-12 | Sol/sağ EAR            | AU45 (Blink)           |
| 13-14 | Sol/sağ göz genişliği  | AU7                    |
| 15    | EAR asimetri           | \|sol - sağ\|          |
| 16    | Ortalama EAR           | (sol + sağ) / 2        |

#### Kaş Action Units (4 özellik)

| #     | Özellik                  | FACS Karşılığı       |
| ----- | ------------------------ | -------------------- |
| 17-18 | Sol/sağ kaş-göz mesafesi | AU1+AU2 (Brow Raise) |
| 19    | İç kaş mesafesi          | AU4 (Brow Lowerer)   |
| 20    | Kaş asimetrisi           | Doğallık ölçüsü      |

#### Yüz Geometri (4 özellik)

| #   | Özellik                    | Amaç              |
| --- | -------------------------- | ----------------- |
| 21  | Burun-çene / yüz genişliği | Yüz uzunluk oranı |
| 22  | Alın-burun / yüz genişliği | Üst yüz oranı     |
| 23  | Toplam yüz oranı           | Yüz şekli         |
| 24  | Dudak-yüz oranı            | Dudak ölçeği      |

### 4.2 DCT Frekans Analizi (10 özellik)

GAN tabanlı deepfake modelleri, frekans düzleminde "checkerboard" artefaktları bırakır. Bu artefaktlar insan gözüyle görülmez ama Discrete Cosine Transform (DCT) ile tespit edilebilir.

**İşlem adımları:**

```
1. Ağız bölgesini kırp (landmark 78, 308, 61, 291, 0, 17)
2. 64×64 piksel'e resize et
3. Gri tonlamaya çevir
4. 2D DCT uygula → frekans matrisi
5. Frekans bölgelerine ayır:
   ┌──────────────────────────────────────────┐
   │ Düşük [0:8]     │ Orta [8:32]           │
   │ (genel yapı)    │ (detaylar)            │
   ├─────────────────┼───────────────────────┤
   │                 │ Yüksek [32:64]        │
   │                 │ (ARTEFAKTLAR!)        │
   └─────────────────┴───────────────────────┘
```

| #   | Özellik                    | Ne tespit eder            |
| --- | -------------------------- | ------------------------- |
| 1-2 | Düşük frekans mean/std     | Genel yapı tutarlılığı    |
| 3-4 | Orta frekans mean/std      | Detay seviyesi            |
| 5-6 | Yüksek frekans mean/std    | GAN artefakt yoğunluğu    |
| 7   | Yüksek/düşük frekans oranı | Artefakt belirginliği     |
| 8   | Orta/düşük frekans oranı   | Detay kalitesi            |
| 9   | Yüksek frekans max         | Artefakt spike            |
| 10  | Outlier sayısı (>2σ)       | Anormal frekans noktaları |

### 4.3 Velocity ve Acceleration (34 + 34 özellik)

Deepfake algoritmaları bazen ağız hareketlerini "yumuşatır" (smoothing), bu da doğal olmayan sabit hızlara neden olur.

```
Velocity (Hız)     = X_t - X_{t-1}        (1. türev)
Acceleration (İvme) = V_t - V_{t-1}       (2. türev)

Gerçek videoda:
  Dudak hızı: 0.02, 0.05, 0.08, 0.03, 0.01  (düzensiz, doğal)

Deepfake videoda:
  Dudak hızı: 0.04, 0.04, 0.04, 0.04, 0.04  (sabit, yapay)
```

Her 34 özellik için (24 AU + 10 DCT) ayrı velocity ve acceleration hesaplanır.

### 4.4 Explicit Senkronizasyon Özellikleri (3 özellik)

Model senkronizasyonu kendi başına öğrenmeye çalışır ama ona yardımcı olmak için **explicit** özellikler verilir:

```
İnsan konuşurken:
  Ses yükselir → dudak açılır  (pozitif korelasyon)
  Ses düşer   → dudak kapanır

Deepfake'te:
  Ses yükselir ↗ ama dudak gecikmeli açılır ‖
  Milisaniyelik kaymalar yaşar
```

| #   | Özellik              | Formül                                | Ne ölçer        |
| --- | -------------------- | ------------------------------------- | --------------- |
| 1   | Pointwise korelasyon | normalize(dudak) × normalize(RMS)     | Anlık uyum      |
| 2   | Uyumsuzluk           | \|normalize(dudak) - normalize(RMS)\| | Anlık fark      |
| 3   | Genişlik-ses         | dudak_genişliği × normalize(RMS)      | Alternatif uyum |

### 4.5 İşitsel Özellikler (20 özellik)

#### MFCC — Mel-Frequency Cepstral Coefficients (13 özellik)

İnsan işitme sistemini taklit eden frekans temsili. Her ses parçası 13 boyutlu vektöre dönüştürülür.

```
Ses Sinyali → FFT → Mel Filtre Bankası → Log → DCT → 13 MFCC katsayısı
```

#### Spectral Özellikler (7 özellik)

| #   | Özellik            | Ne ölçer                          |
| --- | ------------------ | --------------------------------- |
| 14  | Spectral Centroid  | Sesin "ağırlık merkezi" frekansı  |
| 15  | Spectral Rolloff   | Enerjinin %85'inin olduğu frekans |
| 16  | Spectral Bandwidth | Frekans yayılımı                  |
| 17  | Zero Crossing Rate | Ses çalkantısı                    |
| 18  | RMS Energy         | Ses şiddeti                       |
| 19  | Spectral Flatness  | Gürültü vs ton sinyali            |
| 20  | Spectral Contrast  | Frekans tepeleri arası fark       |

### 4.6 Zaman Damgası Senkronizasyonu

Her video karesi ile karşılık gelen ses kesiti eşleştirilir:

```python
samples_per_frame = sr / fps  # 16000 / 30 ≈ 533 ses örneği/kare

Kare #50 için:
  audio_start = 50 × 533 = 26,650
  audio_end   = 26,650 + 533 = 27,183
  ses_kesiti  = audio[26650 : 27183]
```

### 4.7 Toplam Özellik Özeti

```
┌────────────────────────────────────────────────┐
│  GÖRSEL DAL (105 özellik × 30 kare)            │
│  ├── 24  Action Unit mesafesi                  │
│  ├── 10  DCT frekans analizi                   │
│  ├── 34  Velocity (1. türev)                   │
│  ├── 34  Acceleration (2. türev)               │
│  └──  3  Senkronizasyon (dudak × RMS)          │
├────────────────────────────────────────────────┤
│  İŞİTSEL DAL (20 özellik × 30 kare)           │
│  ├── 13  MFCC katsayıları                      │
│  └──  7  Spectral özellikler                   │
├────────────────────────────────────────────────┤
│  TOPLAM: 125 özellik × 30 zaman adımı          │
│  Her örnek tensörü: (30, 105) + (30, 20)       │
└────────────────────────────────────────────────┘
```

---

## 5. Model Mimarisi

### 5.1 Genel Yapı: İki Dallı (Two-Branch) Hibrit

Model, **Functional API** ile tanımlanmıştır. İki ayrı giriş (görsel + işitsel) ayrı dallarda işlenir ve sonunda birleştirilir.

### 5.2 Katman Detayları

#### Conv1D (1 Boyutlu Evrişim)

Zaman serisi üzerinde yerel örüntüleri yakalar. Kernel size=3, yani 3 ardışık kareye bakarak kısa süreli desenleri (dudak açılma/kapanma, göz kırpma) öğrenir.

```
Giriş:  [kare1, kare2, kare3, kare4, kare5]
Kernel:         [w1, w2, w3]
Çıktı:     [c1,    c2,    c3]

c1 = kare1×w1 + kare2×w2 + kare3×w3
c2 = kare2×w1 + kare3×w2 + kare4×w3
```

#### SE-Block (Squeeze-and-Excitation)

Hangi kanalların (özelliklerin) daha önemli olduğunu **öğrenir**. Çok hafif yapı.

```
Giriş: (batch, 30, 64)
  │
  ├─ Squeeze:  Global Average Pool → (batch, 64)
  │    Her kanalın zaman üzerindeki ortalaması
  │
  ├─ Excitation:
  │    Dense(16, relu)  → Sıkıştır
  │    Dense(64, sigmoid) → 0-1 arası ağırlıklar üret
  │
  └─ Scale: Giriş × ağırlıklar
     Önemli kanallar güçlenir, önemsizler zayıflar
```

#### LSTM (Long Short-Term Memory)

Uzun vadeli zamansal bağımlılıkları modeller. 30 karelik pencere boyunca "konuşma ritmi" ve "senkronizasyon tutarlılığı" gibi uzun süreli desenleri yakalar.

```
LSTM Hücre Yapısı:
  ┌──────────┐
  │ Forget Gate │ → Hangi bilgiyi unut
  │ Input Gate  │ → Hangi yeni bilgiyi al
  │ Output Gate │ → Ne çıktı ver
  └──────────┘

return_sequences=True → Her zaman adımı için çıktı üret (Attention için gerekli)
```

#### Self-Attention

LSTM'in 30 karelik çıktısını ağırlıklandırır. "Dudağın kaydığı" kritik 2-3 kareye yüksek ağırlık verir.

```
LSTM çıktısı: [h1, h2, h3, ..., h30]

Attention ağırlıkları hesapla:
  u_i = tanh(W × h_i + b)
  a_i = softmax(u^T × u_i)

Ağırlıklı toplam:
  output = Σ(a_i × h_i)

Örnek:
  Kare  5: a=0.02 (normal)
  Kare 12: a=0.35 (★ dudak kayması tespit!)
  Kare 13: a=0.28 (★ artefakt tespit!)
  Kare 20: a=0.01 (normal)
```

### 5.3 Katman Akışı (Layer-by-Layer)

```
GÖRSEL DAL:
  Input(30, 105)
  │
  Conv1D(64, k=3, relu, L2)     → (30, 64)   Yerel AU değişimleri
  SE-Block(r=4)                  → (30, 64)   Önemli kanalları seç
  BatchNorm                      → (30, 64)   İç kovaryans kaydırma düzelt
  Dropout(0.3)                   → (30, 64)   Overfitting önle
  Conv1D(32, k=3, relu, L2)     → (30, 32)   Daha soyut özellikler
  BatchNorm                      → (30, 32)
  LSTM(64, return_seq=True, L2) → (30, 64)   Zamansal bağımlılıklar
  Dropout(0.3)                   → (30, 64)
  Self-Attention                 → (64,)      Kritik karelere odaklan

İŞİTSEL DAL:
  Input(30, 20)
  │
  Conv1D(64, k=3, relu, L2)     → (30, 64)   Yerel ses değişimleri
  SE-Block(r=4)                  → (30, 64)   Önemli kanalları seç
  BatchNorm                      → (30, 64)
  Dropout(0.3)                   → (30, 64)
  Conv1D(32, k=3, relu, L2)     → (30, 32)   Soyut ses özellikleri
  BatchNorm                      → (30, 32)
  LSTM(64, return_seq=True, L2) → (30, 64)   Ses ritmi, tonlama
  Dropout(0.3)                   → (30, 64)
  Self-Attention                 → (64,)      Kritik ses anlarına odaklan

FÜZYON:
  Concatenate                    → (128,)     İki dalı birleştir
  Dense(128, relu, L2)           → (128,)     Cross-modal etkileşim
  Dropout(0.4)                   → (128,)
  Dense(64, relu, L2)            → (64,)      Daha soyut temsil
  Dropout(0.3)                   → (64,)
  Dense(32, relu)                → (32,)      Son soyutlama
  Dense(1, sigmoid)              → (1,)       Deepfake skoru [0,1]
```

---

## 6. Eğitim Stratejisi

### 6.1 Normalizasyon (Z-Score)

```
X_normalized = (X - mean) / std

Her özellik ayrı ayrı normalize edilir.
Mean ve std eğitim setinden hesaplanır, test setine de uygulanır.
```

### 6.2 Data Augmentation (3× Veri)

```
Orijinal veri:     6,400 örnek
+ Gaussian gürültü: +6,400 (σ=0.03)
+ Downscale sim.:   +6,400 (scale=0.85-1.0)
─────────────────────────────────
Toplam:            19,200 eğitim örneği
```

### 6.3 Regularization (Düzenleme)

| Teknik             | Parametre              | Amaç                           |
| ------------------ | ---------------------- | ------------------------------ |
| L2 Regularization  | λ=0.001                | Ağırlık büyümesini sınırla     |
| Dropout            | 0.3 / 0.4              | Rastgele nöron silme           |
| BatchNormalization | -                      | İç kovaryans kaydırmayı düzelt |
| EarlyStopping      | patience=15            | Overfitting önce dur           |
| ReduceLROnPlateau  | factor=0.5, patience=7 | Plato'da öğrenme hızını düşür  |

### 6.4 Optimizer ve Loss

```
Optimizer:         Adam (lr=0.0005)
Loss Function:     Binary Crossentropy
                   L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
Class Weight:      Otomatik dengeleme
                   w_real = N / (2 × N_real)
                   w_fake = N / (2 × N_fake)
```

---

## 7. Eğitim Sonuçları

### 7.1 Model Evrimi

| #      | Model                              |  Özellik Sayısı  | Val Accuracy |
| ------ | ---------------------------------- | :--------------: | :----------: |
| v1     | LSTM                               |    2+13 = 15     |    %70.4     |
| v2     | Conv1D + LSTM                      |    24+20 = 44    |    %74.1     |
| **v3** | **Conv1D + SE + LSTM + Attention** | **105+20 = 125** |  **%97.3**   |

### 7.2 v3 Eğitim Detayları

```
En iyi epoch:        15
Train Accuracy:      %99.6
Validation Accuracy: %97.3
Validation Loss:     0.2618
Eğitim süresi:       ~7 dakika (CPU)
Early Stopping:      Epoch 30'da (patience=15)
```

---

## 8. Tahmin (Inference) Akışı

```
1. Video okunur (OpenCV)
2. Her 2. kare işlenir (FRAME_SKIP=2)
3. Her kare için:
   a. MediaPipe → 468 landmark → 24 AU mesafesi
   b. Ağız bölgesi kırpılır → DCT → 10 frekans özelliği
   c. Karşılık gelen ses → 13 MFCC + 7 spectral
   d. RMS enerji hesaplanır
4. Tüm kareler üzerinden:
   a. Velocity (1. türev) hesaplanır
   b. Acceleration (2. türev) hesaplanır
   c. Dudak-RMS sync hesaplanır
5. Z-Score normalizasyon uygulanır
6. 30 karelik kayan pencereler oluşturulur
7. Her pencere modele verilir → [0, 1] arası skor
8. Tüm pencerelerin ortalaması → Final Deepfake Skoru

Karar:
  < 0.35  →  ✅ GERÇEK (yüksek güven)
  0.35-0.65 → ⚠️ BELİRSİZ
  > 0.65  →  ❌ SAHTE (yüksek güven)
```

---

## 9. Dosya Yapısı

```
├── dataset_loader.py           Veri seti yükleme + dengeleme + split
│     └─ get_dataset()          → (train_entries, test_entries)
│
├── create_dataset.py           Video → özellik çıkarma pipeline
│     ├─ extract_action_units() → 24 AU vektörü
│     ├─ extract_dct_features() → 10 DCT vektörü
│     ├─ extract_audio_features() → 20 ses vektörü
│     └─ process_single_video() → (vis, aud, rms) dizileri
│
├── train_model.py              Model tanımlama + eğitim
│     ├─ compute_velocity_acceleration() → hız/ivme
│     ├─ compute_sync_features() → dudak-RMS uyum
│     ├─ class SelfAttention    → Dikkat mekanizması
│     ├─ class SEBlock          → Kanal önemi
│     └─ Model: Conv1D+SE+LSTM+Attention+Dense
│
├── predict_video.py            Tek video üzerinde tahmin
│     └─ analyze_video()        → deepfake skoru [0,1]
│
├── egitim_verisi/              İşlenmiş veri (.npy)
│     ├─ X_vis_train/test.npy   Görsel özellikler (34D)
│     ├─ X_aud_train/test.npy   İşitsel özellikler (20D)
│     ├─ X_rms_train/test.npy   RMS enerji
│     ├─ y_train/test.npy       Etiketler
│     └─ vis/aud_mean/std.npy   Normalizasyon parametreleri
│
├── face_landmarker.task        MediaPipe Face Mesh modeli (3.7MB)
├── deepfake_detector_model.h5  Eğitilmiş model
└── deepfake_detector_best.keras En iyi epoch modeli
```

---

## 10. Kütüphane Bağımlılıkları

| Kütüphane        | Amaç                                    |
| ---------------- | --------------------------------------- |
| TensorFlow/Keras | Model mimarisi, eğitim, inference       |
| MediaPipe        | 468 yüz landmark çıkarma (Face Mesh)    |
| OpenCV           | Video kare okuma, renk dönüşümü, resize |
| Librosa          | MFCC, spectral özellikler, ses yükleme  |
| SciPy            | DCT (Discrete Cosine Transform)         |
| NumPy            | Tüm sayısal hesaplamalar                |
| imageio-ffmpeg   | Videodan ses çıkarma                    |
| Matplotlib       | Eğitim grafikleri                       |

---

## 11. Modelin Güçlü ve Zayıf Yönleri

### Güçlü Yönler

- **Hafif mimari**: CNN/Transformer yok, CPU'da çalışır
- **%97.3 accuracy**: FakeAVCeleb veri setinde yüksek başarı
- **Çok modaliteli**: Hem ses hem görüntü aynı anda analiz edilir
- **Frekans analizi**: Piksel seviyesinde inceleme olmadan artefakt tespiti
- **Temporal analiz**: Hız/ivme ile yapay hareket desenleri yakalanır

### Zayıf Yönler

- **Eğitim verisi sınırlı**: Sadece 500 gerçek video
- **Tahmin süresi**: 13 saniyelik video ~5 dakikada analiz ediliyor
- **Domain bağımlılığı**: Farklı kamera/ortam koşullarında performans düşebilir
- **Yüz bağımlılığı**: Yüz bulunamayan karelerde AU=0 vektörü kullanılır
