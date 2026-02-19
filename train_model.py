"""
==================================================================
Sentetik Medya Tespiti - Model Eğitimi (v3)
==================================================================
İyileştirmeler:
1. Self-Attention mekanizması (kritik karelere odaklanma)
2. SE-Block (Squeeze-and-Excitation, kanal önem ağırlıkları)
3. Velocity/Acceleration (landmark hız ve ivme)
4. Explicit senkronizasyon (dudak açıklığı vs RMS korelasyonu)
5. Gelişmiş augmentation (JPEG compression sim., downscale, ses gürültüsü)
6. class_weight ile dengeleme
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Concatenate, Dropout,
    BatchNormalization, Conv1D, Layer, Multiply,
    GlobalAveragePooling1D, Reshape, Permute
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SAVE_DIR = "egitim_verisi"

# ===================== VERİ YÜKLEME =====================
print("[INFO] Eğitim verileri yükleniyor...")
X_vis_train = np.load(f"{SAVE_DIR}/X_vis_train.npy")
X_aud_train = np.load(f"{SAVE_DIR}/X_aud_train.npy")
X_rms_train = np.load(f"{SAVE_DIR}/X_rms_train.npy")
y_train = np.load(f"{SAVE_DIR}/y_train.npy")

X_vis_test = np.load(f"{SAVE_DIR}/X_vis_test.npy")
X_aud_test = np.load(f"{SAVE_DIR}/X_aud_test.npy")
X_rms_test = np.load(f"{SAVE_DIR}/X_rms_test.npy")
y_test = np.load(f"{SAVE_DIR}/y_test.npy")

SEQ_LEN = X_vis_train.shape[1]
N_VIS = X_vis_train.shape[2]   # 34 (24 AU + 10 DCT)
N_AUD = X_aud_train.shape[2]   # 20

# ===================== VELOCITY & ACCELERATION =====================
print("[INFO] Velocity ve acceleration hesaplanıyor...")

def compute_velocity_acceleration(X):
    """Landmark hız (1. türev) ve ivme (2. türev)."""
    velocity = np.diff(X, axis=1)                          # (N, SEQ-1, feat)
    velocity = np.pad(velocity, ((0,0),(1,0),(0,0)))       # Pad to SEQ_LEN
    acceleration = np.diff(velocity, axis=1)
    acceleration = np.pad(acceleration, ((0,0),(1,0),(0,0)))
    return velocity, acceleration

vel_tr, acc_tr = compute_velocity_acceleration(X_vis_train)
vel_te, acc_te = compute_velocity_acceleration(X_vis_test)

# ===================== SYNC FEATURES =====================
print("[INFO] Senkronizasyon özellikleri hesaplanıyor...")

def compute_sync_features(X_vis, X_rms):
    """
    Dudak açıklığı (AU index 0) vs RMS enerji korelasyonu.
    Her pencere için Pearson korelasyonu.
    """
    sync = np.zeros((len(X_vis), SEQ_LEN, 3))
    for i in range(len(X_vis)):
        lip_height = X_vis[i, :, 0]           # AU index 0 = dudak dikey açıklık
        lip_width = X_vis[i, :, 1]            # AU index 1 = dudak genişliği
        rms = X_rms[i, :]                      # RMS enerji
        
        # Frame-wise: |lip - normalized_rms| farkı
        lip_norm = (lip_height - lip_height.mean()) / (lip_height.std() + 1e-8)
        rms_norm = (rms - rms.mean()) / (rms.std() + 1e-8)
        
        sync[i, :, 0] = lip_norm * rms_norm   # Pointwise korelasyon
        sync[i, :, 1] = np.abs(lip_norm - rms_norm)  # Uyumsuzluk
        sync[i, :, 2] = lip_width * rms_norm   # Dudak genişlik-ses ilişkisi
    
    return sync

sync_tr = compute_sync_features(X_vis_train, X_rms_train)
sync_te = compute_sync_features(X_vis_test, X_rms_test)

# ===================== BİRLEŞTİR =====================
# Görsel: 34 AU+DCT + 34 velocity + 34 acceleration + 3 sync = 105
X_vis_train_full = np.concatenate([X_vis_train, vel_tr, acc_tr, sync_tr], axis=2)
X_vis_test_full = np.concatenate([X_vis_test, vel_te, acc_te, sync_te], axis=2)

N_VIS_FULL = X_vis_train_full.shape[2]
print(f"  Görsel: {N_VIS_FULL} özellik (34 AU+DCT + 34 vel + 34 acc + 3 sync)")
print(f"  İşitsel: {N_AUD} özellik")

# ===================== NORMALİZASYON =====================
print("[INFO] Normalize ediliyor...")

vis_mean = X_vis_train_full.reshape(-1, N_VIS_FULL).mean(axis=0)
vis_std = X_vis_train_full.reshape(-1, N_VIS_FULL).std(axis=0) + 1e-8
X_vis_train_full = (X_vis_train_full - vis_mean) / vis_std
X_vis_test_full = (X_vis_test_full - vis_mean) / vis_std

aud_mean = X_aud_train.reshape(-1, N_AUD).mean(axis=0)
aud_std = X_aud_train.reshape(-1, N_AUD).std(axis=0) + 1e-8
X_aud_train = (X_aud_train - aud_mean) / aud_std
X_aud_test = (X_aud_test - aud_mean) / aud_std

np.save(f"{SAVE_DIR}/vis_mean.npy", vis_mean)
np.save(f"{SAVE_DIR}/vis_std.npy", vis_std)
np.save(f"{SAVE_DIR}/aud_mean.npy", aud_mean)
np.save(f"{SAVE_DIR}/aud_std.npy", aud_std)

# NaN temizle
X_vis_train_full = np.nan_to_num(X_vis_train_full, nan=0.0, posinf=0.0, neginf=0.0)
X_vis_test_full = np.nan_to_num(X_vis_test_full, nan=0.0, posinf=0.0, neginf=0.0)
X_aud_train = np.nan_to_num(X_aud_train, nan=0.0, posinf=0.0, neginf=0.0)
X_aud_test = np.nan_to_num(X_aud_test, nan=0.0, posinf=0.0, neginf=0.0)

# ===================== AUGMENTATION =====================
print("[INFO] Data augmentation...")

def augment(X_vis, X_aud, y):
    n = len(y)
    
    # 1. Gaussian gürültü
    noise_v = X_vis + 0.03 * np.random.randn(*X_vis.shape)
    noise_a = X_aud + 0.03 * np.random.randn(*X_aud.shape)
    
    # 2. Downscale simülasyonu (görsel özellikleri hadamard çarpanıyla boz)
    scale = np.random.uniform(0.85, 1.0, size=(n, 1, X_vis.shape[2]))
    scaled_v = X_vis * scale
    scaled_a = X_aud + 0.02 * np.random.randn(*X_aud.shape)
    
    X_vis_aug = np.concatenate([X_vis, noise_v, scaled_v], axis=0)
    X_aud_aug = np.concatenate([X_aud, noise_a, scaled_a], axis=0)
    y_aug = np.concatenate([y, y, y], axis=0)
    
    perm = np.random.permutation(len(y_aug))
    return X_vis_aug[perm], X_aud_aug[perm], y_aug[perm]

X_vis_aug, X_aud_aug, y_aug = augment(X_vis_train_full, X_aud_train, y_train)
print(f"  Train: {len(y_train)} → {len(y_aug)} (x3 augmentation)")

# ===================== CUSTOM LAYERS =====================

class SelfAttention(Layer):
    """
    Self-Attention: Modelin hangi karelerin önemli olduğuna odaklanmasını sağlar.
    Dudağın kaydığı kritik 2-3 kareye ağırlık verir.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def build(self, input_shape):
        self.W = self.add_weight(name='att_W', shape=(input_shape[-1], input_shape[-1]),
                                  initializer='glorot_uniform', trainable=True)
        self.b = self.add_weight(name='att_b', shape=(input_shape[-1],),
                                  initializer='zeros', trainable=True)
        self.u = self.add_weight(name='att_u', shape=(input_shape[-1], 1),
                                  initializer='glorot_uniform', trainable=True)
    
    def call(self, x):
        uit = tf.tanh(tf.matmul(x, self.W) + self.b)
        ait = tf.matmul(uit, self.u)
        ait = tf.squeeze(ait, axis=-1)
        ait = tf.nn.softmax(ait, axis=1)
        ait = tf.expand_dims(ait, axis=-1)
        output = tf.reduce_sum(x * ait, axis=1)
        return output


class SEBlock(Layer):
    """
    Squeeze-and-Excitation Block:
    Hangi kanalların (özelliklerin) daha önemli olduğunu öğrenir.
    Çok hafif yapı (~%1 ekstra parametre).
    """
    def __init__(self, reduction=4, **kwargs):
        super().__init__(**kwargs)
        self.reduction = reduction
    
    def build(self, input_shape):
        channels = input_shape[-1]
        self.fc1 = Dense(channels // self.reduction, activation='relu')
        self.fc2 = Dense(channels, activation='sigmoid')
    
    def call(self, x):
        squeeze = tf.reduce_mean(x, axis=1)        # Global Average Pool
        excitation = self.fc1(squeeze)
        excitation = self.fc2(excitation)
        excitation = tf.expand_dims(excitation, axis=1)
        return x * excitation                        # Channel-wise reweight


# ===================== MODEL =====================
REG = l2(0.001)

# Görsel Dal
input_vis = Input(shape=(SEQ_LEN, N_VIS_FULL), name="Gorsel_Girdi")
v = Conv1D(64, 3, activation='relu', padding='same', kernel_regularizer=REG)(input_vis)
v = SEBlock(reduction=4, name="SE_Gorsel")(v)
v = BatchNormalization()(v)
v = Dropout(0.3)(v)
v = Conv1D(32, 3, activation='relu', padding='same', kernel_regularizer=REG)(v)
v = BatchNormalization()(v)
v = LSTM(64, return_sequences=True, kernel_regularizer=REG)(v)
v = Dropout(0.3)(v)
v = SelfAttention(name="Attention_Gorsel")(v)

# İşitsel Dal
input_aud = Input(shape=(SEQ_LEN, N_AUD), name="Isitsel_Girdi")
a = Conv1D(64, 3, activation='relu', padding='same', kernel_regularizer=REG)(input_aud)
a = SEBlock(reduction=4, name="SE_Isitsel")(a)
a = BatchNormalization()(a)
a = Dropout(0.3)(a)
a = Conv1D(32, 3, activation='relu', padding='same', kernel_regularizer=REG)(a)
a = BatchNormalization()(a)
a = LSTM(64, return_sequences=True, kernel_regularizer=REG)(a)
a = Dropout(0.3)(a)
a = SelfAttention(name="Attention_Isitsel")(a)

# Füzyon
merged = Concatenate(name="Fuzyon")([v, a])
x = Dense(128, activation='relu', kernel_regularizer=REG)(merged)
x = Dropout(0.4)(x)
x = Dense(64, activation='relu', kernel_regularizer=REG)(x)
x = Dropout(0.3)(x)
x = Dense(32, activation='relu')(x)
output = Dense(1, activation='sigmoid', name="Deepfake_Skor")(x)

model = Model(inputs=[input_vis, input_aud], outputs=output,
              name="Deepfake_v3_Attention_SE")

model.compile(optimizer=Adam(learning_rate=0.0005),
              loss='binary_crossentropy', metrics=['accuracy'])

print("\n[MODEL]")
model.summary()

# ===================== EĞİTİM =====================
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True, verbose=1),
    ModelCheckpoint('deepfake_detector_best.keras', monitor='val_accuracy', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=1),
]

# Class weight (dengesizlik telafisi)
n_real = np.sum(y_aug == 0)
n_fake = np.sum(y_aug == 1)
total = len(y_aug)
class_weight = {0: total / (2.0 * n_real), 1: total / (2.0 * n_fake)}
print(f"\n[EĞİTİM] class_weight: {class_weight}")

history = model.fit(
    [X_vis_aug, X_aud_aug], y_aug,
    epochs=100, batch_size=32,
    validation_data=([X_vis_test_full, X_aud_test], y_test),
    callbacks=callbacks,
    class_weight=class_weight,
    verbose=1
)

# ===================== SONUÇLAR =====================
test_loss, test_acc = model.evaluate([X_vis_test_full, X_aud_test], y_test, verbose=0)

print(f"\n{'='*60}")
print(f"  TEST SONUÇLARI")
print(f"  Loss:     {test_loss:.4f}")
print(f"  Accuracy: {test_acc:.4f} ({test_acc*100:.1f}%)")
print(f"{'='*60}")

# Grafikler
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(history.history['accuracy'], label='Train', linewidth=2)
ax1.plot(history.history['val_accuracy'], label='Val', linewidth=2)
ax1.set_title('Accuracy'); ax1.legend(); ax1.grid(True, alpha=0.3)
ax2.plot(history.history['loss'], label='Train', linewidth=2)
ax2.plot(history.history['val_loss'], label='Val', linewidth=2)
ax2.set_title('Loss'); ax2.legend(); ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('training_results.png', dpi=150)
print("[INFO] Grafikler kaydedildi.")

model.save("deepfake_detector_model.h5")
print("[INFO] Model kaydedildi.")