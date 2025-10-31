import os, math, shutil, yaml, glob
from pathlib import Path
from collections import Counter, defaultdict

DATA_YAML = "/content/yolo_seg_dataset/data.yaml"   # <-- change if yours differs

with open(DATA_YAML, "r") as f:
    y = yaml.safe_load(f)

CONVERTED = Path(y["path"])
names = y["names"]
lbl_train = CONVERTED / "labels" / "train"
img_train = CONVERTED / "images" / "train"
assert lbl_train.exists() and img_train.exists(), "train split not found."

# --- 1) Measure imbalance (instance counts per class) ---
class_counts = Counter()
image_classes = {}  # stem -> set of class ids present

for lf in sorted(lbl_train.glob("*.txt")):
    present = set()
    with open(lf, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cid = int(line.split()[0])
            class_counts[cid] += 1
            present.add(cid)
    image_classes[lf.stem] = present

# report
print("\nPer-class instance counts (train):")
for cid, name in enumerate(names):
    print(f"{cid:2d}  {name:<24}  {class_counts[cid]}")

max_count = max(class_counts.values())
print(f"\nLargest class has {max_count} instances.")


MAX_MULT = 5  # <= raise carefully if your dataset is tiny

img_exts = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]
def find_image_for_stem(stem):
    for ext in img_exts:
        p = img_train / f"{stem}{ext}"
        if p.exists():
            return p
    return None

bal_img_root = CONVERTED / "images" / "train_balanced"
bal_lbl_root = CONVERTED / "labels" / "train_balanced"
for d in (bal_img_root, bal_lbl_root):
    d.mkdir(parents=True, exist_ok=True)

# copy originals first (so every sample appears at least once)
for stem in image_classes.keys():
    src_img = find_image_for_stem(stem)
    src_lbl = lbl_train / f"{stem}.txt"
    assert src_img and src_lbl.exists(), f"missing {stem}"
    shutil.copy2(src_img, bal_img_root / src_img.name)
    shutil.copy2(src_lbl, bal_lbl_root / src_lbl.name)

# now duplicate according to need
deficit = {c: max(1, math.ceil(max_count / max(1, class_counts[c]))) for c in range(len(names))}
for stem, cls_set in image_classes.items():
    need = max(deficit[c] for c in cls_set)  # how many total copies desired
    need = min(MAX_MULT, need)
    if need <= 1:
        continue
    src_img = find_image_for_stem(stem)
    src_lbl = lbl_train / f"{stem}.txt"
    for k in range(1, need):  # we already copied the original once
        img_out = bal_img_root / f"{stem}_dup{k}{src_img.suffix}"
        lbl_out = bal_lbl_root / f"{stem}_dup{k}.txt"
        shutil.copy2(src_img, img_out)
        shutil.copy2(src_lbl, lbl_out)

# quick sanity: recount in balanced view (by instances, approximated via labels)
bal_counts = Counter()
for lf in sorted(bal_lbl_root.glob("*.txt")):
    with open(lf, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            bal_counts[int(line.split()[0])] += 1

print("\nBalanced-view instance counts (capped by MAX_MULT):")
for cid, name in enumerate(names):
    print(f"{cid:2d}  {name:<24}  {bal_counts[cid]}")

# --- 3) Point training to train_balanced ---
y["train"] = "images/train_balanced"
with open(DATA_YAML, "w") as f:
    yaml.safe_dump(y, f, sort_keys=False)
print(f"\nUpdated {DATA_YAML} → now uses images/train_balanced for training.")
