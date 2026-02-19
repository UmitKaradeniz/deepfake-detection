"""
FakeAVCeleb v1.2 Dataset Loader (v2)
======================================
Asimetrik örnekleme: 500 real + 2000 fake
class_weight ile dengeleme model tarafında yapılır.
"""

import csv
import os
import random
import numpy as np

DATASET_ROOT = r"C:\Users\dkara\Desktop\FakeAVCeleb_v1.2\FakeAVCeleb_v1.2"
CSV_PATH = os.path.join(DATASET_ROOT, "meta_data.csv")

REAL_SAMPLES = 500       # Tüm real videoları kullan
FAKE_SAMPLES = 500       # Dengeli: Real ile aynı sayıda
TEST_RATIO = 0.2
RANDOM_SEED = 42


def load_metadata():
    """meta_data.csv dosyasını okur."""
    entries = []
    with open(CSV_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_type = row['type'].strip()
            video_filename = row['path'].strip()
            folder_path_csv = row.get('', '').strip()

            if folder_path_csv:
                relative_folder = folder_path_csv.replace("FakeAVCeleb/", "", 1)
                full_path = os.path.join(DATASET_ROOT, relative_folder, video_filename)
            else:
                race = row.get('race', '').strip()
                gender = row.get('gender', '').strip()
                source = row.get('source', '').strip()
                full_path = os.path.join(DATASET_ROOT, video_type, race, gender, source, video_filename)

            label = 0 if video_type == "RealVideo-RealAudio" else 1

            entries.append({
                'path': full_path,
                'type': video_type,
                'label': label,
                'method': row.get('method', '').strip(),
                'source': row.get('source', '').strip(),
            })
    return entries


def filter_existing_videos(entries):
    valid = [e for e in entries if os.path.exists(e['path'])]
    missing = len(entries) - len(valid)
    if missing > 0:
        print(f"  UYARI: {missing} video bulunamadı, atlanıyor.")
    return valid


def create_subset(entries, n_real=REAL_SAMPLES, n_fake=FAKE_SAMPLES, seed=RANDOM_SEED):
    """Asimetrik alt küme: Tüm real + n_fake fake."""
    random.seed(seed)

    real_videos = [e for e in entries if e['label'] == 0]
    fake_videos = [e for e in entries if e['label'] == 1]

    print(f"  Toplam Real: {len(real_videos)}, Toplam Fake: {len(fake_videos)}")

    n_r = min(n_real, len(real_videos))
    n_f = min(n_fake, len(fake_videos))

    selected_real = random.sample(real_videos, n_r)
    selected_fake = random.sample(fake_videos, n_f)

    result = selected_real + selected_fake
    random.shuffle(result)

    print(f"  Seçilen: {n_r} Real + {n_f} Fake = {len(result)} video")
    return result


def train_test_split(entries, test_ratio=TEST_RATIO, seed=RANDOM_SEED):
    random.seed(seed)
    real = [e for e in entries if e['label'] == 0]
    fake = [e for e in entries if e['label'] == 1]

    def split(lst, ratio):
        random.shuffle(lst)
        n = int(len(lst) * ratio)
        return lst[n:], lst[:n]

    tr_r, te_r = split(real, test_ratio)
    tr_f, te_f = split(fake, test_ratio)

    train = tr_r + tr_f
    test = te_r + te_f
    random.shuffle(train)
    random.shuffle(test)
    return train, test


def get_dataset(n_real=REAL_SAMPLES, n_fake=FAKE_SAMPLES):
    """Ana fonksiyon."""
    print("FakeAVCeleb v1.2 yükleniyor...")
    all_entries = load_metadata()
    print(f"  CSV: {len(all_entries)} kayıt")

    valid = filter_existing_videos(all_entries)
    print(f"  Geçerli: {len(valid)} video")

    subset = create_subset(valid, n_real, n_fake)
    train, test = train_test_split(subset)

    print(f"  Train: {len(train)} | Test: {len(test)}")
    print(f"  Train - Real: {sum(1 for e in train if e['label']==0)} | Fake: {sum(1 for e in train if e['label']==1)}")
    print(f"  Test  - Real: {sum(1 for e in test if e['label']==0)} | Fake: {sum(1 for e in test if e['label']==1)}")

    return train, test


if __name__ == "__main__":
    train_data, test_data = get_dataset()
