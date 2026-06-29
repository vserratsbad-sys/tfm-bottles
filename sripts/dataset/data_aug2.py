"""
Dataset Augmentation Script para línea de embotellado - YOLO format
=====================================================================
⚠️  ORDEN CORRECTO DEL PIPELINE:
        1º  ds_split.py   → parte las 400 ORIGINALES
        2º  data_aug.py   (ESTE) → aumenta SOLO images/train

    Este script trabaja IN PLACE sobre la carpeta de TRAIN ya partida.
    Nunca toca val ni test, de modo que el conjunto de test sigue siendo
    completamente independiente (sin fuga por augmentación).

Orden de aplicación (todo dentro de train):
  1. Iluminación simulada (3 tipos)  → +100% sobre total inicial   (33/33/34%)
  2. Desplazamiento horizontal suave → +50%  sobre total tras paso 1
  3. Desenfoque de movimiento horiz  → +25%  sobre total tras paso 2

NUNCA se sobreescriben imágenes: siempre se añaden con sufijo único.
"""

import os
import cv2
import numpy as np
import random
import math
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIGURACIÓN — cambia solo aquí
# ─────────────────────────────────────────────
DATASET_DIR = Path(__file__).resolve().parent.parent / "data" / "DS_FINAL"

# ⚠️  Apunta al TRAIN ya partido. NO a la carpeta plana de originales.
IMAGES_SUBDIR = "images/train"      # subcarpeta de imágenes (train)
LABELS_SUBDIR = "labels/train"      # subcarpeta de labels YOLO (train)

# Porcentajes por fase (respecto al total en ese momento)
PCT_ILUMINACION  = 1.00   # +100% del total inicial (las imágenes de train)
PCT_SPLIT_LUZ    = [0.33, 0.33, 0.34]   # reparto entre los 3 tipos (deben sumar 1.0)

PCT_DESPL        = 0.50   # +50% del total tras iluminación
PCT_BLUR         = 0.25   # +25% del total tras desplazamiento

# Parámetros de augmentación
SHIFT_MAX_PCT    = 0.05   # desplazamiento máximo: 5% del ancho de imagen (suave)
BLUR_KERNEL_RANGE = (2, 3)  # rango de longitud del kernel motion-blur (píxeles)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ─────────────────────────────────────────────
# UTILIDADES
# ─────────────────────────────────────────────

def get_pairs(img_dir: Path, lbl_dir: Path):
    """Devuelve lista de (img_path, lbl_path) para los pares existentes."""
    pairs = []
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if lbl_path.exists():
            pairs.append((img_path, lbl_path))
    return pairs


def unique_stem(img_dir: Path, base_stem: str) -> str:
    """Genera un stem único añadiendo sufijo numérico si ya existe."""
    candidate = base_stem
    counter = 0
    while (img_dir / (candidate + ".jpg")).exists():
        counter += 1
        candidate = f"{base_stem}_{counter:04d}"
    return candidate


def save_pair(img: np.ndarray, labels: list[str],
              img_dir: Path, lbl_dir: Path, stem: str):
    """Guarda imagen + label con el stem dado (nunca sobreescribe)."""
    final_stem = unique_stem(img_dir, stem)
    cv2.imwrite(str(img_dir / (final_stem + ".jpg")), img)
    with open(lbl_dir / (final_stem + ".txt"), "w") as f:
        f.write("\n".join(labels))
    return final_stem


def read_label(lbl_path: Path) -> list[str]:
    with open(lbl_path) as f:
        return [l.rstrip() for l in f if l.strip()]

# ─────────────────────────────────────────────
# PASO 1 — Iluminación simulada
# ─────────────────────────────────────────────

def aug_luz_brillante(img: np.ndarray) -> np.ndarray:
    """Simula sobreiluminación / luz directa intensa."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    factor = random.uniform(1.1, 1.4)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def aug_luz_tenue(img: np.ndarray) -> np.ndarray:
    """Simula luz tenue / zona de sombra en la línea."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    factor = random.uniform(0.60, 0.85)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def aug_luz_lateral(img: np.ndarray) -> np.ndarray:
    """Simula gradiente de iluminación lateral (foco desde un lado)."""
    h, w = img.shape[:2]
    # Gradiente horizontal: más luz en un lado, menos en el otro
    lado = random.choice(["izq", "der"])
    gradient = np.linspace(0.4, 1.6, w) if lado == "izq" else np.linspace(1.6, 0.4, w)
    gradient = gradient.astype(np.float32)
    mask = np.tile(gradient, (h, 1))          # (H, W)
    mask = np.stack([mask] * 3, axis=-1)      # (H, W, 3)
    result = np.clip(img.astype(np.float32) * mask, 0, 255).astype(np.uint8)
    return result


LUZ_FUNCS = [aug_luz_brillante, aug_luz_tenue, aug_luz_lateral]
LUZ_NOMBRES = ["brillante", "tenue", "lateral"]


def fase_iluminacion(pairs, img_dir, lbl_dir, pct_total, splits):
    n_total = len(pairs)
    n_nuevas = round(n_total * pct_total)   # p.ej. 280 * 1.0 = 280

    # Calcular cuántas por cada tipo
    counts = [math.floor(n_nuevas * s) for s in splits]
    counts[-1] = n_nuevas - sum(counts[:-1])   # el último absorbe el redondeo

    print(f"\n[FASE 1 — ILUMINACIÓN]")
    print(f"  Total actual      : {n_total}")
    print(f"  Nuevas a generar  : {n_nuevas}  ({counts} por tipo)")

    creadas = 0
    for tipo_idx, (func, nombre, count) in enumerate(zip(LUZ_FUNCS, LUZ_NOMBRES, counts)):
        elegidas = random.choices(pairs, k=count)
        for img_path, lbl_path in elegidas:
            img = cv2.imread(str(img_path))
            labels = read_label(lbl_path)
            aug = func(img)
            stem = f"{img_path.stem}_luz_{nombre}"
            save_pair(aug, labels, img_dir, lbl_dir, stem)
            creadas += 1
        print(f"  · luz_{nombre:<10}: {count} imágenes generadas")

    print(f"  ✓ Total creadas   : {creadas}")

# ─────────────────────────────────────────────
# PASO 2 — Desplazamiento horizontal suave
# ─────────────────────────────────────────────

def shift_labels_x(labels: list[str], dx_norm: float, direction: str) -> list[str]:
    """
    Ajusta las coordenadas x_center de los labels YOLO tras desplazamiento.
    dx_norm: fracción del ancho desplazada (positivo = derecha).
    Descarta boxes que queden fuera de [0,1].
    """
    nuevas = []
    for line in labels:
        parts = line.split()
        if len(parts) < 5:
            nuevas.append(line)
            continue
        cls = parts[0]
        xc, yc, bw, bh = map(float, parts[1:5])
        xc_new = xc + dx_norm
        # Recortar al borde
        x_min = xc_new - bw / 2
        x_max = xc_new + bw / 2
        if x_max <= 0 or x_min >= 1:
            continue   # box completamente fuera, la descartamos
        # Ajustar si está parcialmente fuera
        x_min_clip = max(0.0, x_min)
        x_max_clip = min(1.0, x_max)
        bw_new = x_max_clip - x_min_clip
        xc_clip = (x_min_clip + x_max_clip) / 2
        nuevas.append(f"{cls} {xc_clip:.6f} {yc:.6f} {bw_new:.6f} {bh:.6f}")
    return nuevas


def aug_shift(img: np.ndarray, labels: list[str], max_pct: float):
    """Desplazamiento horizontal suave (izq o der), devuelve (img_aug, labels_aug, sufijo)."""
    h, w = img.shape[:2]
    direction = random.choice(["izq", "der"])
    dx_pct = random.uniform(0.02, max_pct)   # mínimo 2% para que sea visible
    dx_px = int(w * dx_pct)
    dx_norm = dx_pct if direction == "der" else -dx_pct

    M = np.float32([[1, 0, dx_px if direction == "der" else -dx_px], [0, 1, 0]])
    aug = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    labels_aug = shift_labels_x(labels, dx_norm, direction)
    sufijo = f"shift_{direction}_{int(dx_pct*100)}pct"
    return aug, labels_aug, sufijo


def fase_desplazamiento(pairs, img_dir, lbl_dir, pct):
    n_total = len(pairs)
    n_nuevas = round(n_total * pct)

    print(f"\n[FASE 2 — DESPLAZAMIENTO HORIZONTAL]")
    print(f"  Total actual      : {n_total}")
    print(f"  Nuevas a generar  : {n_nuevas}  (+{int(pct*100)}%)")

    elegidas = random.choices(pairs, k=n_nuevas)
    for img_path, lbl_path in elegidas:
        img = cv2.imread(str(img_path))
        labels = read_label(lbl_path)
        aug, labels_aug, sufijo = aug_shift(img, labels, SHIFT_MAX_PCT)
        stem = f"{img_path.stem}_{sufijo}"
        save_pair(aug, labels_aug, img_dir, lbl_dir, stem)

    print(f"  ✓ Total creadas   : {n_nuevas}")

# ─────────────────────────────────────────────
# PASO 3 — Desenfoque de movimiento horizontal
# ─────────────────────────────────────────────

def motion_blur_horizontal(img: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Aplica motion blur horizontal hacia la derecha.
    Simula la estela de una botella moviéndose rápido por la línea.
    """
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1.0 / kernel_size
    return cv2.filter2D(img, -1, kernel)


def fase_motion_blur(pairs, img_dir, lbl_dir, pct):
    n_total = len(pairs)
    n_nuevas = round(n_total * pct)

    print(f"\n[FASE 3 — DESENFOQUE DE MOVIMIENTO (línea embotellado)]")
    print(f"  Total actual      : {n_total}")
    print(f"  Nuevas a generar  : {n_nuevas}  (+{int(pct*100)}%)")
    print(f"  Kernel blur rango : {BLUR_KERNEL_RANGE[0]}–{BLUR_KERNEL_RANGE[1]} px (horizontal →)")

    elegidas = random.choices(pairs, k=n_nuevas)
    for img_path, lbl_path in elegidas:
        img = cv2.imread(str(img_path))
        labels = read_label(lbl_path)
        k = random.randrange(BLUR_KERNEL_RANGE[0], BLUR_KERNEL_RANGE[1] + 1, 2)  # impar
        if k % 2 == 0:
            k += 1
        aug = motion_blur_horizontal(img, k)
        stem = f"{img_path.stem}_mblur_k{k}"
        save_pair(aug, labels, img_dir, lbl_dir, stem)

    print(f"  ✓ Total creadas   : {n_nuevas}")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    img_dir = Path(DATASET_DIR) / IMAGES_SUBDIR
    lbl_dir = Path(DATASET_DIR) / LABELS_SUBDIR

    assert img_dir.exists(), (
        f"No existe: {img_dir}\n"
        f"   ¿Has ejecutado primero ds_split.py? Este script aumenta SOLO el train ya partido."
    )
    assert lbl_dir.exists(), f"No existe: {lbl_dir}"

    print("=" * 55)
    print(" AUGMENTACIÓN DE DATASET — LÍNEA DE EMBOTELLADO")
    print(" (SOLO sobre train — val y test NO se tocan)")
    print("=" * 55)
    print(f" Carpeta objetivo: {img_dir}")

    # ── FASE 1: Iluminación ──────────────────────────────
    pairs_0 = get_pairs(img_dir, lbl_dir)
    print(f"\nTrain inicial: {len(pairs_0)} pares imagen+label")

    fase_iluminacion(pairs_0, img_dir, lbl_dir, PCT_ILUMINACION, PCT_SPLIT_LUZ)

    # ── FASE 2: Desplazamiento ───────────────────────────
    pairs_1 = get_pairs(img_dir, lbl_dir)
    fase_desplazamiento(pairs_1, img_dir, lbl_dir, PCT_DESPL)

    # ── FASE 3: Motion Blur ──────────────────────────────
    pairs_2 = get_pairs(img_dir, lbl_dir)
    fase_motion_blur(pairs_2, img_dir, lbl_dir, PCT_BLUR)

    # ── RESUMEN ──────────────────────────────────────────
    pairs_final = get_pairs(img_dir, lbl_dir)
    print("\n" + "=" * 55)
    print(" RESUMEN FINAL (train)")
    print("=" * 55)
    print(f"  Fase 0 (train original): {len(pairs_0):>5} imágenes")
    print(f"  Tras Fase 1 (luz)      : {len(pairs_1):>5} imágenes  (+{len(pairs_1)-len(pairs_0)})")
    print(f"  Tras Fase 2 (shift)    : {len(pairs_2):>5} imágenes  (+{len(pairs_2)-len(pairs_1)})")
    print(f"  Tras Fase 3 (blur)     : {len(pairs_final):>5} imágenes  (+{len(pairs_final)-len(pairs_2)})")
    print(f"\n  ✓ Train aumentado de {len(pairs_0)} → {len(pairs_final)} imágenes")
    print(f"  ✓ val y test intactos (solo originales) → test independiente")
    print("=" * 55)


if __name__ == "__main__":
    main()
