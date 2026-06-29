from pathlib import Path
import cv2

labels_dir = Path("runs/inferall/exp/labels")
images_dir = Path("runs/inferall/exp/img")
output_dir = Path("scripts/ocr/crops")

output_dir.mkdir(parents=True, exist_ok=True)

for label_file in labels_dir.rglob("*.txt"):

    # Buscar imagen correspondiente
    image_file = None
    for ext in [".jpg", ".jpeg", ".png"]:
        candidate = images_dir / label_file.relative_to(labels_dir).with_suffix(ext)
        if candidate.exists():
            image_file = candidate
            break

    if image_file is None:
        print(f"No encontrada imagen para {label_file}")
        continue

    img = cv2.imread(str(image_file))
    h, w = img.shape[:2]

    with open(label_file, "r") as f:
        lines = f.readlines()

    crop_idx = 0

    for line in lines:
        parts = line.strip().split()

        if not parts:
            continue

        cls = int(parts[0])

        # Solo clase 4
        if cls != 4:
            continue

        x_center, y_center, bw, bh = map(float, parts[1:5])  # ignora conf si existe

        x1 = int((x_center - bw / 2) * w)
        y1 = int((y_center - bh / 2) * h)
        x2 = int((x_center + bw / 2) * w)
        y2 = int((y_center + bh / 2) * h)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        crop = img[y1:y2, x1:x2]

        output_name = f"{image_file.stem}_class4_{crop_idx}.jpg"
        cv2.imwrite(str(output_dir / output_name), crop)

        crop_idx += 1

print("Terminado.")