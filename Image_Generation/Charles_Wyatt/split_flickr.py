import os 
import random
import shutil


ROOT = 'data/flickr/resized/'

TRAIN_DIR = os.path.join(ROOT, 'train/img')
VAL_DIR = os.path.join(ROOT, 'val/img')

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

file = [
    f for f in os.listdir(ROOT) 
    if f.lower().endswith(('.png', '.jpg', '.jpeg')) and os.path.isfile(os.path.join(ROOT, f))
]

random.shuffle(file)

split_idx = int(0.8 * len(file))
train_files = file[:split_idx]
val_files = file[split_idx:]

for f in train_files:
    shutil.move(os.path.join(ROOT, f), os.path.join(TRAIN_DIR, f))

for f in val_files:
    shutil.move(os.path.join(ROOT, f), os.path.join(VAL_DIR, f))

print(f"Done! {len(train_files)} train |{len(val_files)} val")