# TFM — Sistema piloto de inspección visual basado en YOLO26n + OCR

Máster Universitario en Inteligencia Artificial — UNIR  
Autor: Valentín Serrats Badia

---

## Contenido

- `scripts/dataset/` — partición y aumento de datos
- `scripts/model/` — optimización de hiperparámetros, entrenamiento e inferencia
- `scripts/ocr/` — pipeline de OCR para verificación de trazabilidad
- `checkpoints/best.pt` — modelo entrenado final (YOLO26n)
- `data.yaml` — configuración del dataset
- `requirements.txt` — dependencias

## Dataset

El dataset no se incluye en este repositorio por su tamaño.  
Disponible en: https://drive.google.com/drive/folders/1BnRllOPYl-O5rY5_1Eu74lUy2uE59NgH?usp=sharing

Contiene dos carpetas:
- `DS_FINAL/` — dataset completo utilizado para el entrenamiento, con la partición train/val/test ya aplicada y el subconjunto de entrenamiento aumentado (1.050 imágenes)
- `DSA_backup/` — las 400 imágenes originales sin aumentar, antes de aplicar ningún proceso de partición ni augmentation

## Nota sobre rutas

Los scripts contienen rutas configuradas para el entorno de desarrollo original.  
Antes de ejecutarlos es necesario adaptarlas al entorno local.

## Requisitos

```bash
pip install -r requirements.txt
```