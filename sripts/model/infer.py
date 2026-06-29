"""
infer.py  —  Inferencia con el modelo entrenado
Uso:
    python infer.py --source foto.jpg
    python infer.py --source carpeta/imagenes/
    python infer.py --source 0          # webcam
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
from pathlib import Path

MODEL  = "checkpoints/best.pt"
IMGSZ  = 640
CONF   = 0.25   # umbral de confianza (ajusta a tu gusto)
IOU    = 0.45   # umbral NMS

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True,
                        help="Ruta a imagen, carpeta, vídeo o índice de webcam")
    parser.add_argument("--model",  default=MODEL)
    parser.add_argument("--conf",   type=float, default=CONF)
    parser.add_argument("--iou",    type=float, default=IOU)
    parser.add_argument("--imgsz",  type=int,   default=IMGSZ)
    parser.add_argument("--device", default="0",
                        help="0=GPU, cpu=CPU")
    parser.add_argument("--save",   action="store_true", default=True,
                        help="Guarda imagen con bounding boxes")
    args = parser.parse_args()

    model = YOLO(args.model)

    results = model.predict(
        source   = args.source,
        imgsz    = args.imgsz,
        conf     = args.conf,
        iou      = args.iou,
        device   = args.device,
        save     = args.save,
        save_txt = True,
        save_conf= True,
        verbose  = True,
        project  = str(Path(__file__).resolve().parent / "runs/inferall"),  # ← añade esto
        name     = "exp",                        # ← y esto
    )

    # Resumen por imagen
    for r in results:
        print(f"\n📷  {r.path}")
        if len(r.boxes) == 0:
            print("   Sin detecciones.")
        else:
            for box in r.boxes:
                cls_id = int(box.cls)
                name   = model.names[cls_id]
                conf   = float(box.conf)
                xyxy   = box.xyxy[0].tolist()
                print(f"   [{name}]  conf={conf:.2f}  bbox={[round(v,1) for v in xyxy]}")

    print(f"\n✅ Resultados guardados en runs/predict/")

if __name__ == "__main__":
    main()