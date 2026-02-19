"""
==================================================================
Sentetik Medya Tespiti - Gelişmiş Veri Seti Oluşturucu (v3)
==================================================================
Yeni özellikler:
1. 24 Action Unit mesafesi (v2 ile aynı)
2. DCT frekans analizi (yüz bölgesi, +10 özellik)
3. Velocity/Acceleration (landmark hız ve ivme)
4. Explicit senkronizasyon (dudak vs RMS korelasyonu)
5. 20 ses özelliği (13 MFCC + 7 spectral)
6. Gelişmiş augmentation (JPEG compression, downscale)
"""

import cv2
import mediapipe as mp
import librosa
import numpy as np
import math
import os
import subprocess
import tempfile
import time
from scipy.fft import dct

import imageio_ffmpeg
from dataset_loader import get_dataset

# ===================== AYARLAR =====================
SAVE_DIR = "egitim_verisi"
N_MFCC = 13
MAX_DURATION_SEC = 3
SEQUENCE_LENGTH = 30
FRAME_SKIP = 2

# Özellik boyutları
NUM_AU_FEATURES = 24        # Action Unit mesafeleri
NUM_DCT_FEATURES = 10       # DCT frekans özellikleri
NUM_AUDIO_FEATURES = 20     # 13 MFCC + 7 spectral
# Velocity (24) + Acceleration (24) → train_model.py'de hesaplanır
# Sync features → train_model.py'de hesaplanır

NUM_VIS_TOTAL = NUM_AU_FEATURES + NUM_DCT_FEATURES  # 34
NUM_AUD_TOTAL = NUM_AUDIO_FEATURES                    # 20

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# ===================== MEDIAPIPE =====================
model_path = 'face_landmarker.task'
options = mp.tasks.vision.FaceLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
)

print("[INFO] MediaPipe Face Mesh yükleniyor...")
landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
print("[INFO] Hazır.")

ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()


# ===================== GÖRSEL ÖZELLİKLER =====================
def dist(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)


def extract_action_units(lm):
    """24 AU özelliği (önceki versiyon ile aynı)."""
    fw = dist(lm[234], lm[454]) + 1e-8
    lo = dist(lm[13], lm[14])
    lw = dist(lm[78], lm[308])
    leh = dist(lm[159], lm[145])
    reh = dist(lm[386], lm[374])
    lew = dist(lm[33], lm[133])
    rew = dist(lm[362], lm[263])
    lear = leh / (lew + 1e-8)
    rear = reh / (rew + 1e-8)

    return [
        lo, lw, lo / (lw + 1e-8), dist(lm[0], lm[17]),
        dist(lm[61], lm[291]), dist(lm[82], lm[312]),
        dist(lm[87], lm[317]), dist(lm[152], lm[6]) / fw,
        leh, reh, lear, rear,
        lew, rew, abs(lear - rear), (lear + rear) / 2,
        dist(lm[70], lm[159]) / fw, dist(lm[300], lm[386]) / fw,
        dist(lm[107], lm[336]) / fw,
        abs(dist(lm[70], lm[159]) - dist(lm[300], lm[386])) / fw,
        dist(lm[1], lm[152]) / fw, dist(lm[10], lm[1]) / fw,
        dist(lm[10], lm[152]) / fw, lw / fw,
    ]


def extract_dct_features(frame_rgb, landmarks, target_size=64):
    """
    DCT Frekans Analizi:
    Yüz bölgesinden (özellikle ağız çevresi) kırpılmış 64x64 patch'e
    2D DCT uygular. GAN'ların bıraktığı checkerboard artefaktlarını yakalar.
    
    Returns: 10 frekans özelliği
    """
    h, w = frame_rgb.shape[:2]
    
    # Ağız bölgesini kırp (landmark 0=üst dudak, 17=alt dudak, 78=sol, 308=sağ)
    lm = landmarks
    x_coords = [int(lm[i].x * w) for i in [78, 308, 61, 291, 0, 17]]
    y_coords = [int(lm[i].y * h) for i in [78, 308, 61, 291, 0, 17]]
    
    x_min = max(0, min(x_coords) - 20)
    x_max = min(w, max(x_coords) + 20)
    y_min = max(0, min(y_coords) - 20)
    y_max = min(h, max(y_coords) + 20)
    
    if x_max - x_min < 10 or y_max - y_min < 10:
        return [0.0] * NUM_DCT_FEATURES
    
    # Gri tonlama ve resize
    mouth_patch = frame_rgb[y_min:y_max, x_min:x_max]
    gray = cv2.cvtColor(mouth_patch, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray = cv2.resize(gray, (target_size, target_size))
    
    # 2D DCT uygula
    dct_2d = dct(dct(gray, axis=0, norm='ortho'), axis=1, norm='ortho')
    
    # Frekans bölgelerinden özellikler çıkar
    low = dct_2d[:8, :8]                  # Düşük frekans (genel yapı)
    mid = dct_2d[8:32, 8:32]              # Orta frekans (detaylar)
    high = dct_2d[32:, 32:]               # Yüksek frekans (artefaktlar)
    
    features = [
        np.mean(np.abs(low)),              # 1. Düşük frekans ortalama
        np.std(np.abs(low)),               # 2. Düşük frekans std
        np.mean(np.abs(mid)),              # 3. Orta frekans ortalama
        np.std(np.abs(mid)),               # 4. Orta frekans std
        np.mean(np.abs(high)),             # 5. Yüksek frekans ortalama
        np.std(np.abs(high)),              # 6. Yüksek frekans std
        np.mean(np.abs(high)) / (np.mean(np.abs(low)) + 1e-8),  # 7. Yüksek/düşük oran
        np.mean(np.abs(mid)) / (np.mean(np.abs(low)) + 1e-8),   # 8. Orta/düşük oran
        np.max(np.abs(high)),              # 9. Yüksek frekans max (artifact spike)
        np.sum(np.abs(high) > np.mean(np.abs(high)) + 2*np.std(np.abs(high))),  # 10. Outlier sayısı
    ]
    
    return features


# ===================== SES ÖZELLİKLER =====================
def extract_audio_features(audio_slice, sr):
    """20 ses özelliği: 13 MFCC + 7 spectral."""
    if len(audio_slice) < 512:
        audio_slice = np.pad(audio_slice, (0, 512 - len(audio_slice)))

    mfcc = librosa.feature.mfcc(y=audio_slice, sr=sr, n_mfcc=N_MFCC, n_fft=512, hop_length=256)
    mfcc_mean = np.mean(mfcc.T, axis=0)

    sc = np.mean(librosa.feature.spectral_centroid(y=audio_slice, sr=sr, n_fft=512))
    sr_f = np.mean(librosa.feature.spectral_rolloff(y=audio_slice, sr=sr, n_fft=512))
    sb = np.mean(librosa.feature.spectral_bandwidth(y=audio_slice, sr=sr, n_fft=512))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio_slice))
    rms_val = np.mean(librosa.feature.rms(y=audio_slice))
    sf = np.mean(librosa.feature.spectral_flatness(y=audio_slice))
    try:
        scon = np.mean(librosa.feature.spectral_contrast(y=audio_slice, sr=sr, n_fft=512, n_bands=3))
    except:
        scon = 0.0

    return np.concatenate([mfcc_mean, [sc, sr_f, sb, zcr, rms_val, sf, scon]])


def extract_rms_energy(audio_slice):
    """Ses RMS enerjisi (senkronizasyon için ayrıca döndürülür)."""
    if len(audio_slice) < 512:
        audio_slice = np.pad(audio_slice, (0, 512 - len(audio_slice)))
    return float(np.sqrt(np.mean(audio_slice**2)))


# ===================== VİDEO İŞLEME =====================
def extract_audio_from_video(video_path):
    temp_audio = os.path.join(tempfile.gettempdir(), f"temp_{os.getpid()}.wav")
    cmd = [ffmpeg_path, '-y', '-i', video_path, '-vn', '-acodec', 'pcm_s16le',
           '-ar', '16000', '-ac', '1', '-loglevel', 'error', temp_audio]
    result = subprocess.run(cmd, capture_output=True)
    return temp_audio if result.returncode == 0 else None


def process_single_video(video_path):
    """
    Tek video → görsel (34D) + işitsel (20D) + RMS dizisi
    Velocity ve sync train_model.py'de hesaplanır.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None, None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        return None, None, None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_f = min(total, int(fps * MAX_DURATION_SEC))

    temp_audio = extract_audio_from_video(video_path)
    if temp_audio is None:
        cap.release()
        return None, None, None

    try:
        y, sr = librosa.load(temp_audio, sr=16000, duration=MAX_DURATION_SEC)
    except:
        cap.release()
        if os.path.exists(temp_audio): os.remove(temp_audio)
        return None, None, None

    if os.path.exists(temp_audio): os.remove(temp_audio)

    spf = int(sr / fps)
    chunk = spf * FRAME_SKIP
    if chunk <= 0:
        cap.release()
        return None, None, None

    vis_list, aud_list, rms_list = [], [], []
    idx = 0

    while cap.isOpened() and idx < max_f:
        ok, img = cap.read()
        if not ok: break
        if idx % FRAME_SKIP != 0:
            idx += 1
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        try:
            res = landmarker.detect(mp_img)
        except:
            idx += 1
            continue

        # AU + DCT
        au = [0.0] * NUM_AU_FEATURES
        dct_feat = [0.0] * NUM_DCT_FEATURES
        if res.face_landmarks:
            try:
                au = extract_action_units(res.face_landmarks[0])
                dct_feat = extract_dct_features(rgb, res.face_landmarks[0])
            except:
                pass

        vis_vector = au + dct_feat  # 24 + 10 = 34

        # Ses
        start = idx * spf
        end = start + chunk
        aslice = y[start:end] if end <= len(y) else np.zeros(chunk)
        
        aud_vector = extract_audio_features(aslice, sr)
        rms_val = extract_rms_energy(aslice)

        vis_list.append(vis_vector)
        aud_list.append(aud_vector)
        rms_list.append(rms_val)
        idx += 1

    cap.release()

    if len(vis_list) < SEQUENCE_LENGTH:
        return None, None, None

    return np.array(vis_list), np.array(aud_list), np.array(rms_list)


def create_sequences(vis, aud, rms, label):
    X_v, X_a, X_r, Y = [], [], [], []
    for i in range(len(vis) - SEQUENCE_LENGTH):
        X_v.append(vis[i:i + SEQUENCE_LENGTH])
        X_a.append(aud[i:i + SEQUENCE_LENGTH])
        X_r.append(rms[i:i + SEQUENCE_LENGTH])
        Y.append(label)
    return X_v, X_a, X_r, Y


def process_list(entries, desc=""):
    all_v, all_a, all_r, all_y = [], [], [], []
    ok, skip = 0, 0
    t0 = time.time()

    for i, e in enumerate(entries):
        v, a, r = process_single_video(e['path'])
        if v is None:
            skip += 1
        else:
            sv, sa, sr_, sy = create_sequences(v, a, r, e['label'])
            all_v.extend(sv)
            all_a.extend(sa)
            all_r.extend(sr_)
            all_y.extend(sy)
            ok += 1

        if (i + 1) % 20 == 0 or i == len(entries) - 1:
            el = time.time() - t0
            sp = (i + 1) / el if el > 0 else 0
            rem = (len(entries) - i - 1) / sp if sp > 0 else 0
            print(f"  [{desc}] {i+1}/{len(entries)} | OK:{ok} Skip:{skip} | "
                  f"{sp:.1f}v/s | ~{rem:.0f}s")

    print(f"  [{desc}] Bitti: {ok} video → {len(all_v)} pencere")

    if not all_v:
        return None, None, None, None
    return np.array(all_v), np.array(all_a), np.array(all_r), np.array(all_y)


# ===================== ANA =====================
if __name__ == "__main__":
    print("=" * 60)
    print("  Veri Seti Oluşturucu v3 (DCT + Sync + Velocity)")
    print(f"  Görsel: {NUM_VIS_TOTAL} = {NUM_AU_FEATURES} AU + {NUM_DCT_FEATURES} DCT")
    print(f"  İşitsel: {NUM_AUD_TOTAL} = {N_MFCC} MFCC + 7 Spectral")
    print(f"  + RMS (senkronizasyon için)")
    print("=" * 60)

    train_e, test_e = get_dataset()

    print(f"\n--- TRAIN ({len(train_e)} video) ---")
    Xv_tr, Xa_tr, Xr_tr, y_tr = process_list(train_e, "TRAIN")

    print(f"\n--- TEST ({len(test_e)} video) ---")
    Xv_te, Xa_te, Xr_te, y_te = process_list(test_e, "TEST")

    if Xv_tr is not None and Xv_te is not None:
        np.save(f"{SAVE_DIR}/X_vis_train.npy", Xv_tr)
        np.save(f"{SAVE_DIR}/X_aud_train.npy", Xa_tr)
        np.save(f"{SAVE_DIR}/X_rms_train.npy", Xr_tr)
        np.save(f"{SAVE_DIR}/y_train.npy", y_tr)
        np.save(f"{SAVE_DIR}/X_vis_test.npy", Xv_te)
        np.save(f"{SAVE_DIR}/X_aud_test.npy", Xa_te)
        np.save(f"{SAVE_DIR}/X_rms_test.npy", Xr_te)
        np.save(f"{SAVE_DIR}/y_test.npy", y_te)

        print(f"\n{'='*60}")
        print(f"  HAZIR!")
        print(f"  Vis: {Xv_tr.shape} | Aud: {Xa_tr.shape} | RMS: {Xr_tr.shape}")
        print(f"  Train: {len(y_tr)} | Test: {len(y_te)}")
        print(f"  Train - Real: {np.sum(y_tr==0)} | Fake: {np.sum(y_tr==1)}")
        print(f"{'='*60}")
    else:
        print("[HATA] Yeterli veri yok!")

    landmarker.close()