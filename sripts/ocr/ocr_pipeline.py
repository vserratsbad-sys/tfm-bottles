import cv2
import numpy as np
import os
from paddleocr import PaddleOCR

import time


# =========================================================
# OCR ENGINE (GLOBAL)
# =========================================================
ocr = PaddleOCR(
    use_angle_cls=True,
    lang="en",
    use_gpu=True  # <- añadir esto
)


# =========================================================
# PREPROCESSING
# =========================================================
def preprocess_image(img, scale=3):

    img = cv2.resize(
        img,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    gray = clahe.apply(gray)

    # volver a BGR (IMPORTANTE PARA PADDLEOCR)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# =========================================================
# ROTATION (no crop loss)
# =========================================================
def rotate_image(image, angle):

    h, w = image.shape[:2]
    center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos = abs(M[0, 0])
    sin = abs(M[0, 1])

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    return cv2.warpAffine(
        image,
        M,
        (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )


# =========================================================
# OCR WRAPPER
# =========================================================
def run_ocr(img):

    result = ocr.ocr(img, cls=True)

    texts = []
    confs = []

    if result and result[0]:
        for line in result[0]:
            texts.append(line[1][0])
            confs.append(float(line[1][1]))

    if len(confs) == 0:
        return {"text": "", "confidence": 0.0}

    return {
        "text": "\n".join(texts),
        "confidence": float(np.mean(confs))
    }


# =========================================================
# MULTI-ANGLE OCR
# =========================================================
def ocr_multi_angle(img, angles=None):

    if angles is None:
        angles = [0, 45, -45, 70, -70]

    # Primero probar sin rotación
    res0 = run_ocr(img)
    best = {
        "text": res0["text"],
        "confidence": res0["confidence"],
        "angle": 0,
        "rotated_img": img
    }

    # Solo explorar otros ángulos si la confianza es baja
    if best["confidence"] < 0.7:
        for angle in angles[1:]:  # saltar el 0
            rotated = rotate_image(img, angle)
            res = run_ocr(rotated)
            if res["confidence"] > best["confidence"]:
                best = {
                    "text": res["text"],
                    "confidence": res["confidence"],
                    "angle": angle,
                    "rotated_img": rotated
                }

    return best



# =========================================================
# RESULT PANEL (VISUAL)
# =========================================================
def create_result_panel(text, confidence, width=400):

    panel = np.zeros((400, width, 3), dtype=np.uint8)

    lines = text.split("\n") if text else ["(no text detected)"]

    y = 40

    for line in lines:

        cv2.putText(
            panel,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

        y += 35

    cv2.putText(
        panel,
        f"CONF: {confidence:.3f}",
        (10, 370),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
        cv2.LINE_AA
    )

    return panel


def visualize_pipeline(img, binary_clean, binary_dilated, contours, angle, rotated, result):

    # --- 1. Original ---
    vis_orig = img.copy()

    # --- 2. Binarizada limpia ---
    vis_binary = cv2.cvtColor(binary_clean, cv2.COLOR_GRAY2BGR)

    # --- 3. Dilatada + contornos + línea de orientación ---
    vis_contours = cv2.cvtColor(binary_dilated, cv2.COLOR_GRAY2BGR)

    for c in contours:
        if cv2.contourArea(c) < 50:
            continue
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        cv2.drawContours(vis_contours, [box], 0, (0, 255, 0), 2)

        # Línea en la dirección LARGA del objeto
        center = tuple(np.int32(rect[0]))
        w, h = rect[1]
        a = rect[2]
        if w < h:
            a = a + 90
        length = max(w, h) / 2
        dx = int(np.cos(np.radians(a)) * length)
        dy = int(np.sin(np.radians(a)) * length)
        cv2.arrowedLine(vis_contours,
                        (center[0] - dx, center[1] - dy),
                        (center[0] + dx, center[1] + dy),
                        (0, 0, 255), 2, tipLength=0.2)

    # Ángulo en la imagen
    cv2.putText(vis_contours, f"{angle:.2f} grad", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # --- 4. Rotada ---
    vis_rotated = rotated.copy()

    # --- 5. Panel resultado ---
    vis_panel = create_result_panel(result["text"], result["confidence"], width=400)

    # --- Unir todo en una fila ---
    def to_bgr(im):
        if len(im.shape) == 2:
            return cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        return im

    # Escalar todas a la misma altura
    target_h = max(img.shape[0], vis_rotated.shape[0], 400)

    def resize_to_h(im, h):
        im = to_bgr(im)
        ratio = h / im.shape[0]
        return cv2.resize(im, (int(im.shape[1] * ratio), h))

    row = np.hstack([
        resize_to_h(vis_orig, target_h),
        resize_to_h(vis_binary, target_h),
        resize_to_h(vis_contours, target_h),
        resize_to_h(vis_rotated, target_h),
        resize_to_h(vis_panel, target_h),
    ])

    # Etiquetas encima
    labels = ["ORIGINAL", "BINARIA", "DETECCION", "ROTADA", "RESULTADO"]
    label_row = np.zeros((30, row.shape[1], 3), dtype=np.uint8)
    x = 0
    for i, (label, im) in enumerate(zip(labels, [vis_orig, vis_binary, vis_contours, vis_rotated, vis_panel])):
        w = int(im.shape[1] * target_h / im.shape[0]) if im.shape[0] != target_h else im.shape[1]
        cv2.putText(label_row, label, (x + 5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        x += w

    return np.vstack([label_row, row])


def detect_text_angle(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary_clean = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)

    contours, _ = cv2.findContours(binary_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > 50]

    from collections import Counter

    if not contours:
        return 0, binary_clean, binary_clean, contours

    angles = []

    for c in contours:

        rect = cv2.minAreaRect(c)
        angle = rect[-1]
        w, h = rect[1]

        # El lado largo es la dirección del texto
        if w < h:
            angle += 90

        angles.append(angle)

    # ===== VOTACIÓN =====

    bin_size = 5  # grados

    # Cada ángulo vota por el bin más cercano
    angle_bins = [
        round(a / bin_size) * bin_size
        for a in angles
    ]

    # Bin ganador
    dominant_bin = Counter(angle_bins).most_common(1)[0][0]

    # Ángulos que pertenecen al bin ganador
    selected_angles = [
        a for a in angles
        if abs(a - dominant_bin) < bin_size
    ]

    # Ángulo final refinado
    angle = np.mean(selected_angles)

    angle = angle % 180
    if angle > 90:
        angle -= 180

    # === DEBUG TEMPORAL ===
    print("---- DEBUG ANGULOS POR CONTORNO ----", flush = True)
    for c in contours:
        rect = cv2.minAreaRect(c)
        raw_angle = rect[-1]
        w, h = rect[1]
        area = cv2.contourArea(c)
        adj_angle = raw_angle + 90 if w < h else raw_angle
        print(f"area={area:7.1f}  w={w:6.1f} h={h:6.1f}  raw_angle={raw_angle:7.2f}  adj_angle={adj_angle:7.2f}", flush = True)
    print("---- FIN DEBUG ----", flush = True)
    # === FIN DEBUG TEMPORAL ===

    return angle, binary_clean, binary_clean, contours


def process_ocr_pipeline(img):

    angle, binary_clean, binary_dilated, contours = detect_text_angle(img)
    print(f"[DEBUG] angle usado para rotar = {angle:.2f}", flush=True)
    print(f"[DEBUG] -angle (lo que se pasa a rotate_image) = {-angle:.2f}", flush=True)
    rotated = rotate_image(img, angle)
    result = run_ocr(rotated)
    result["angle"] = angle
    result["rotated_img"] = rotated
    result["debug_img"] = visualize_pipeline(
        img, binary_clean, binary_dilated, contours, angle, rotated, result
    )
    return result



# =========================================================
# CONCAT IMAGE + PANEL
# =========================================================
def concat_image_and_panel(img, rotated, panel):

    h = max(img.shape[0], rotated.shape[0], panel.shape[0])

    def pad(im):
        if len(im.shape) == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        out = np.zeros((h, im.shape[1], 3), dtype=np.uint8)
        out[:im.shape[0], :im.shape[1]] = im
        return out

    return np.hstack([pad(img), pad(rotated), pad(panel)])


# =========================================================
# BATCH INFERENCE
# =========================================================
def run_folder_inference(input_folder, output_folder, pipeline_fn):

    os.makedirs(output_folder, exist_ok=True)

    files = [
        f for f in os.listdir(input_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    times = []

    for f in files:
        path = os.path.join(input_folder, f)
        img = cv2.imread(path)

        t0 = time.perf_counter()
        result = pipeline_fn(img)
        elapsed = time.perf_counter() - t0

        times.append(elapsed)
        print(f"[TIMING] {f}: {elapsed:.3f}s", flush=True)

        panel = create_result_panel(result["text"], result["confidence"])
        out = concat_image_and_panel(img, result["rotated_img"], panel)
        cv2.imwrite(os.path.join(output_folder, f), result["debug_img"])

    if times:
        print("\nTODOS LOS TIEMPOS:")
        for i, t in enumerate(times, 1):
            print(f"Imagen {i}: {t:.3f}s")

        print(f"\n[TIMING] Imágenes procesadas : {len(times)}")
        print(f"[TIMING] Tiempo total        : {sum(times):.3f}s")
        print(f"[TIMING] Tiempo promedio     : {sum(times)/len(times):.3f}s")
        print(f"[TIMING] Tiempo mínimo       : {min(times):.3f}s")
        print(f"[TIMING] Tiempo máximo       : {max(times):.3f}s")