"""
Split dataset YOLO  →  train / val / test
==========================================
⚠️  ORDEN CORRECTO DEL PIPELINE:
        1º  ds_split.py   (ESTE)  → parte las 400 ORIGINALES
        2º  data_aug.py           → aumenta SOLO images/train

    Si aumentas antes de partir, las versiones aumentadas de una misma
    imagen se reparten entre train/val/test y el test deja de ser
    independiente (métricas infladas). Por eso se parte primero.

Lee  (SOLO ORIGINALES, las 400):
    DSA/img/   ← imágenes
    DSA/lbl/   ← labels YOLO (.txt)

Genera:
    DSA/images/train/   DSA/images/val/   DSA/images/test/
    DSA/labels/train/   DSA/labels/val/   DSA/labels/test/

Uso:
    python ds_split.py
"""

import shutil
import random
from pathlib import Path

# ─────────────────────────────────────────────
#  CONFIGURACIÓN — cambia solo la ruta base
# ─────────────────────────────────────────────
DSA_ROOT = Path(__file__).resolve().parent.parent / "data" / "DS_FINAL"

SRC_IMG  = DSA_ROOT / "img"      # ← aquí están las 400 originales
SRC_LBL  = DSA_ROOT / "lbl"
DST_ROOT = DSA_ROOT             # las carpetas images/ y labels/ se crean aquí

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15   # lo que sobre va a test

SEED = 42
# ─────────────────────────────────────────────

assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6, \
    "Los ratios deben sumar 1.0"

random.seed(SEED)

# ── Recoger pares imagen + label ─────────────
EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
pairs = []
for img in sorted(SRC_IMG.iterdir()):
    if img.suffix.lower() not in EXTS:
        continue
    lbl = SRC_LBL / (img.stem + ".txt")
    if not lbl.exists():
        print(f"  ⚠️  Sin label: {img.name} — ignorado")
        continue
    pairs.append((img, lbl))

if not pairs:
    print("❌ No se encontraron pares imagen+label.")
    exit(1)

# ── SALVAGUARDA: no debe haber imágenes aumentadas entre las originales ──
# Si esto salta, significa que estás partiendo datos ya aumentados → fuga.
AUG_MARKERS = ("_luz_", "_shift_", "_mblur")
flagged = [img.name for img, _ in pairs if any(m in img.stem for m in AUG_MARKERS)]
if flagged:
    print("❌ ABORTADO: hay imágenes que parecen AUMENTADAS en la carpeta de originales:")
    for name in flagged[:10]:
        print(f"     · {name}")
    if len(flagged) > 10:
        print(f"     · ... y {len(flagged) - 10} más")
    print("   La carpeta de originales (img/) debe contener SOLO las 400 originales.")
    print("   Aumenta DESPUÉS de partir, con data_aug.py sobre images/train.")
    exit(1)

print(f"✅ Pares encontrados (solo originales): {len(pairs)}")

# ── Mezclar y dividir ─────────────────────────
random.shuffle(pairs)
n       = len(pairs)
n_train = int(n * TRAIN_RATIO)
n_val   = int(n * VAL_RATIO)

splits = {
    "train": pairs[:n_train],
    "val":   pairs[n_train:n_train + n_val],
    "test":  pairs[n_train + n_val:],
}

# ── Copiar archivos ───────────────────────────
for split, items in splits.items():
    img_dir = DST_ROOT / "images" / split
    lbl_dir = DST_ROOT / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for img, lbl in items:
        shutil.copy2(img, img_dir / img.name)
        shutil.copy2(lbl, lbl_dir / lbl.name)

    print(f"  {split:<6} → {len(items):>4} imágenes")

# ── Confirmación metodológica (útil para la memoria) ──
stems = {s: {img.stem for img, _ in items} for s, items in splits.items()}
overlap = (stems["train"] & stems["val"]) | (stems["train"] & stems["test"]) | (stems["val"] & stems["test"])
if overlap:
    print(f"❌ Solapamiento entre splits: {overlap}")
else:
    print("\n✓ Sin solapamiento: val y test contienen solo originales NUNCA vistas en train.")
    print("  La augmentación se aplicará únicamente a train → test independiente.")

print(f"\n📁 Dataset listo en: {DST_ROOT.resolve()}")
print("""
Estructura generada:
  DSA/
  ├── images/
  │   ├── train/   ← se aumentará con data_aug.py
  │   ├── val/     ← SOLO originales, NO tocar
  │   └── test/    ← SOLO originales, NO tocar
  └── labels/
      ├── train/
      ├── val/
      └── test/

Siguiente paso:  python data_aug.py   (apunta ya a images/train)
""")
