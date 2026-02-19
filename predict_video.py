"""
==================================================================
Sentetik Medya Tespiti - Video Tahmin v3
==================================================================
DCT frekans analizi, velocity, acceleration ve 
senkronizasyon özellikleriyle geliştirilmiş tahmin.
"""

import cv2
import mediapipe as mp
import librosa
import numpy as np
import math
import os
import sys
import subprocess
import tempfile
import tensorflow as tf
from tensorflow.keras.models import load_model
from scipy.fft import dct
import imageio_ffmpeg

# Özel katmanları kaydet (model yüklemek için gerekli)
from train_model import SelfAttention, SEBlock

MODEL_PATH = "deepfake_detector_model.h5"
VIDEO_PATH = "ses-analiz-test-1.mp4"
SAVE_DIR = "egitim_verisi"
SEQUENCE_LENGTH = 30
FRAME_SKIP = 2
N_MFCC = 13
NUM_AU = 24
NUM_DCT = 10
NUM_VIS = NUM_AU + NUM_DCT  # 34
NUM_AUD = 20

if len(sys.argv) > 1:
    VIDEO_PATH = sys.argv[1]

# ===================== MODEL =====================
print(f"[INFO] Model yükleniyor: {MODEL_PATH}")
model = load_model(MODEL_PATH, custom_objects={
    'SelfAttention': SelfAttention, 'SEBlock': SEBlock
})
print("[INFO] Model hazır.")

# Normalizasyon
try:
    vis_mean = np.load(f"{SAVE_DIR}/vis_mean.npy")
    vis_std = np.load(f"{SAVE_DIR}/vis_std.npy")
    aud_mean = np.load(f"{SAVE_DIR}/aud_mean.npy")
    aud_std = np.load(f"{SAVE_DIR}/aud_std.npy")
    USE_NORM = True
except:
    USE_NORM = False
    print("[UYARI] Normalizasyon parametreleri bulunamadı.")

# ===================== MEDIAPIPE =====================
model_path_mp = 'face_landmarker.task'
options = mp.tasks.vision.FaceLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=model_path_mp),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_faces=1, min_face_detection_confidence=0.5, min_face_presence_confidence=0.5,
)
landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()


# ===================== ÖZELLİK FONKSIYONLARI =====================
def dist(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)


def extract_action_units(lm):
    fw = dist(lm[234], lm[454]) + 1e-8
    lo = dist(lm[13], lm[14]); lw = dist(lm[78], lm[308])
    leh = dist(lm[159], lm[145]); reh = dist(lm[386], lm[374])
    lew = dist(lm[33], lm[133]); rew = dist(lm[362], lm[263])
    lear = leh / (lew + 1e-8); rear = reh / (rew + 1e-8)
    return [
        lo, lw, lo/(lw+1e-8), dist(lm[0],lm[17]),
        dist(lm[61],lm[291]), dist(lm[82],lm[312]),
        dist(lm[87],lm[317]), dist(lm[152],lm[6])/fw,
        leh, reh, lear, rear, lew, rew,
        abs(lear-rear), (lear+rear)/2,
        dist(lm[70],lm[159])/fw, dist(lm[300],lm[386])/fw,
        dist(lm[107],lm[336])/fw,
        abs(dist(lm[70],lm[159])-dist(lm[300],lm[386]))/fw,
        dist(lm[1],lm[152])/fw, dist(lm[10],lm[1])/fw,
        dist(lm[10],lm[152])/fw, lw/fw,
    ]


def extract_dct_features(frame_rgb, lm, size=64):
    h, w = frame_rgb.shape[:2]
    xc = [int(lm[i].x*w) for i in [78,308,61,291,0,17]]
    yc = [int(lm[i].y*h) for i in [78,308,61,291,0,17]]
    x1, x2 = max(0,min(xc)-20), min(w,max(xc)+20)
    y1, y2 = max(0,min(yc)-20), min(h,max(yc)+20)
    if x2-x1 < 10 or y2-y1 < 10: return [0.0]*NUM_DCT
    patch = cv2.cvtColor(frame_rgb[y1:y2,x1:x2], cv2.COLOR_RGB2GRAY).astype(np.float32)
    patch = cv2.resize(patch, (size, size))
    d = dct(dct(patch, axis=0, norm='ortho'), axis=1, norm='ortho')
    lo = d[:8,:8]; mi = d[8:32,8:32]; hi = d[32:,32:]
    return [
        np.mean(np.abs(lo)), np.std(np.abs(lo)),
        np.mean(np.abs(mi)), np.std(np.abs(mi)),
        np.mean(np.abs(hi)), np.std(np.abs(hi)),
        np.mean(np.abs(hi))/(np.mean(np.abs(lo))+1e-8),
        np.mean(np.abs(mi))/(np.mean(np.abs(lo))+1e-8),
        np.max(np.abs(hi)),
        np.sum(np.abs(hi) > np.mean(np.abs(hi))+2*np.std(np.abs(hi))),
    ]


def extract_audio_features(audio_slice, sr):
    if len(audio_slice) < 512:
        audio_slice = np.pad(audio_slice, (0, 512-len(audio_slice)))
    mfcc = librosa.feature.mfcc(y=audio_slice, sr=sr, n_mfcc=N_MFCC, n_fft=512, hop_length=256)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    sc = np.mean(librosa.feature.spectral_centroid(y=audio_slice, sr=sr, n_fft=512))
    sr_f = np.mean(librosa.feature.spectral_rolloff(y=audio_slice, sr=sr, n_fft=512))
    sb = np.mean(librosa.feature.spectral_bandwidth(y=audio_slice, sr=sr, n_fft=512))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio_slice))
    rms = np.mean(librosa.feature.rms(y=audio_slice))
    sf = np.mean(librosa.feature.spectral_flatness(y=audio_slice))
    try: scon = np.mean(librosa.feature.spectral_contrast(y=audio_slice, sr=sr, n_fft=512, n_bands=3))
    except: scon = 0.0
    return np.concatenate([mfcc_mean, [sc, sr_f, sb, zcr, rms, sf, scon]])


# ===================== ANALİZ =====================
def analyze_video(video_path):
    print(f"\n[ANALİZ] {os.path.basename(video_path)}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): print("[HATA] Video açılamadı!"); return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = total/fps if fps > 0 else 0
    print(f"  {int(cap.get(3))}x{int(cap.get(4))} | {fps:.0f}fps | {dur:.1f}s")
    
    temp = os.path.join(tempfile.gettempdir(), "temp_pred.wav")
    subprocess.run([ffmpeg_path, '-y', '-i', video_path, '-vn', '-acodec', 'pcm_s16le',
                    '-ar', '16000', '-ac', '1', '-loglevel', 'error', temp], capture_output=True)
    try:
        y, sr = librosa.load(temp, sr=16000)
        os.remove(temp)
    except: return None
    
    spf = int(sr/fps); chunk = spf * FRAME_SKIP
    vis_list, aud_list, rms_list = [], [], []
    idx, faces = 0, 0
    
    print("  İşleniyor...", end="", flush=True)
    while cap.isOpened():
        ok, img = cap.read()
        if not ok: break
        if idx % FRAME_SKIP != 0: idx += 1; continue
        
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        try: res = landmarker.detect(mp_img)
        except: idx += 1; continue
        
        au = [0.0]*NUM_AU; dct_f = [0.0]*NUM_DCT
        if res.face_landmarks:
            try:
                au = extract_action_units(res.face_landmarks[0])
                dct_f = extract_dct_features(rgb, res.face_landmarks[0])
                faces += 1
            except: pass
        
        start = idx*spf; end = start+chunk
        aslice = y[start:end] if end <= len(y) else np.zeros(chunk)
        
        vis_list.append(au + dct_f)
        aud_list.append(extract_audio_features(aslice, sr))
        rms_list.append(float(np.sqrt(np.mean(aslice**2))))
        idx += 1
    
    cap.release()
    print(f" OK ({len(vis_list)} kare, {faces} yüz)")
    
    if len(vis_list) < SEQUENCE_LENGTH:
        print("[HATA] Video çok kısa!"); return None
    
    X_vis = np.array(vis_list)  # (T, 34)
    X_aud = np.array(aud_list)  # (T, 20)
    X_rms = np.array(rms_list)  # (T,)
    
    # Velocity & Acceleration (train_model ile aynı)
    vel = np.diff(X_vis, axis=0)
    vel = np.pad(vel, ((1,0),(0,0)))
    acc = np.diff(vel, axis=0)
    acc = np.pad(acc, ((1,0),(0,0)))
    
    # Sync
    lip_h = X_vis[:, 0]
    lip_w = X_vis[:, 1]
    ln = (lip_h - lip_h.mean()) / (lip_h.std() + 1e-8)
    rn = (X_rms - X_rms.mean()) / (X_rms.std() + 1e-8)
    sync = np.stack([ln*rn, np.abs(ln-rn), lip_w*rn], axis=1)  # (T, 3)
    
    # Birleştir: 34 + 34 + 34 + 3 = 105
    X_vis_full = np.concatenate([X_vis, vel, acc, sync], axis=1)
    
    # Normalizasyon
    if USE_NORM:
        X_vis_full = (X_vis_full - vis_mean) / vis_std
        X_aud = (X_aud - aud_mean) / aud_std
    
    X_vis_full = np.nan_to_num(X_vis_full, nan=0.0, posinf=0.0, neginf=0.0)
    X_aud = np.nan_to_num(X_aud, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Sliding window
    iv, ia = [], []
    for i in range(len(X_vis_full) - SEQUENCE_LENGTH):
        iv.append(X_vis_full[i:i+SEQUENCE_LENGTH])
        ia.append(X_aud[i:i+SEQUENCE_LENGTH])
    
    iv = np.array(iv); ia = np.array(ia)
    print(f"  Pencere: {len(iv)}")
    
    preds = model.predict([iv, ia], verbose=0)
    return float(np.mean(preds))


# ===================== SONUÇ =====================
score = analyze_video(VIDEO_PATH)

if score is not None:
    print(f"\n{'='*60}")
    print(f"  DEEPFAKE ANALİZ SONUCU")
    print(f"{'='*60}")
    print(f"  Video:  {os.path.basename(VIDEO_PATH)}")
    print(f"  Skor:   {score:.4f}")
    print(f"  Güven:  {abs(score - 0.5)*200:.1f}%")
    print(f"{'='*60}")
    
    if score > 0.65:
        print(f"  SONUÇ: ❌ SAHTE (DEEPFAKE) TESPİT EDİLDİ")
        print(f"    → Dudak-ses senkronizasyonunda tutarsızlık")
        print(f"    → Frekans düzleminde artefakt tespit edildi")
    elif score < 0.35:
        print(f"  SONUÇ: ✅ GERÇEK VİDEO")
        print(f"    → İşitsel-görsel tutarlılık sağlanıyor")
        print(f"    → Davranışsal biyometri deseni normal")
    else:
        print(f"  SONUÇ: ⚠️  BELİRSİZ (Düşük Güven)")
        print(f"    → Model kesin karar veremiyor")
        print(f"    → Farklı koşullarda yeniden test önerilir")
    
    print(f"{'='*60}")

landmarker.close()