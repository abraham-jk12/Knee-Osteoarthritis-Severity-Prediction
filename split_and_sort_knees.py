"""
Split bilateral knee X-rays into individual knees and sort by KL grade.

- LEFT half of image  = Patient's RIGHT knee = SIDE=1
- RIGHT half of image = Patient's LEFT knee  = SIDE=2

KL grade source: kxr_sq_bu01.txt
- SIDE=1 → V01XRKL = KL grade for RIGHT knee
- SIDE=2 → V01XRKL = KL grade for LEFT knee
"""

import os
import pandas as pd
from pathlib import Path
from PIL import Image

KL_FILE    = r"C:\Users\jkall\Downloads\OAICompleteData_ASCII\OAICompleteData_ASCII\kxr_sq_bu01.txt"
IMAGE_ROOT = r"C:\Users\jkall\Downloads\Package_1244889\results\P001"
OUTPUT_ROOT = r"C:\Users\jkall\Downloads\KL_Sorted_Split"

print("Loading KL grade data...")
df = pd.read_csv(KL_FILE, sep='|')
df = df[['ID', 'SIDE', 'V01XRKL']].copy()
df['ID'] = df['ID'].astype(str).str.strip()
df['SIDE'] = df['SIDE'].astype(int)
df['V01XRKL'] = pd.to_numeric(df['V01XRKL'], errors='coerce')
df = df.dropna(subset=['V01XRKL'])
df['V01XRKL'] = df['V01XRKL'].astype(int)
df = df[df['V01XRKL'].between(0, 4)]

print(f"Loaded {len(df)} KL grade records")
print(f"SIDE=1 (Right knee) records: {len(df[df['SIDE']==1])}")
print(f"SIDE=2 (Left knee) records:  {len(df[df['SIDE']==2])}")

kl_lookup = {}
for _, row in df.iterrows():
    kl_lookup[(str(row['ID']), int(row['SIDE']))] = int(row['V01XRKL'])

print(f"\nKL lookup entries: {len(kl_lookup)}")

for kl in range(5):
    os.makedirs(os.path.join(OUTPUT_ROOT, f"KL{kl}"), exist_ok=True)
print(f"Created output folders in: {OUTPUT_ROOT}")

copied = 0
skipped_no_match = 0
skipped_error = 0

image_root = Path(IMAGE_ROOT)

for view_folder in sorted(image_root.iterdir()):      # 0.C.2, 0.E.1
    if not view_folder.is_dir():
        continue
    print(f"\nProcessing view folder: {view_folder.name}")

    for patient_folder in sorted(view_folder.iterdir()):   # patient ID
        if not patient_folder.is_dir():
            continue
        patient_id = patient_folder.name.strip()

        for date_folder in sorted(patient_folder.iterdir()):   # visit date
            if not date_folder.is_dir():
                continue

            for file in sorted(date_folder.iterdir()):
                if file.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
                    continue

                try:
                    img = Image.open(file)
                    w, h = img.size

                    # Skip if image is too small
                    if w < 200 or h < 200:
                        continue

                    # Split image down the middle 
                    mid = w // 2

                    #LEFT half of image = Patient's RIGHT knee = SIDE=1
                    right_knee_img = img.crop((0, 0, mid, h))

                    #RIGHT half of image = Patient's LEFT knee = SIDE=2
                    left_knee_img = img.crop((mid, 0, w, h))

                    #Save RIGHT knee (SIDE=1, left half of image)
                    kl_right = kl_lookup.get((patient_id, 1))
                    if kl_right is not None:
                        dest = os.path.join(
                            OUTPUT_ROOT,
                            f"KL{kl_right}",
                            f"{patient_id}_{view_folder.name}_RIGHT_{file.stem}.jpg"
                        )
                        right_knee_img.save(dest, quality=95)
                        copied += 1
                    else:
                        skipped_no_match += 1

                    #Save LEFT knee (SIDE=2, right half of image)
                    kl_left = kl_lookup.get((patient_id, 2))
                    if kl_left is not None:
                        dest = os.path.join(
                            OUTPUT_ROOT,
                            f"KL{kl_left}",
                            f"{patient_id}_{view_folder.name}_LEFT_{file.stem}.jpg"
                        )
                        left_knee_img.save(dest, quality=95)
                        copied += 1
                    else:
                        skipped_no_match += 1

                except Exception as e:
                    print(f"  Error processing {file}: {e}")
                    skipped_error += 1

print("\n" + "="*50)
print(f"Done!")
print(f"Total knee images saved: {copied}")
print(f"Skipped (no KL match):   {skipped_no_match}")
print(f"Skipped (errors):        {skipped_error}")
print("\nClass distribution:")
for kl in range(5):
    folder = Path(OUTPUT_ROOT) / f"KL{kl}"
    count = len(list(folder.iterdir()))
    print(f"  KL{kl}: {count} images")
