import os
import cv2
from pathlib import Path

from ocr_pipeline import (
    process_ocr_pipeline,
    run_folder_inference
)


# -----------------------------
# MAIN CONFIG
# -----------------------------
INPUT_FOLDER = Path(__file__).resolve().parent / "crops"
OUTPUT_FOLDER = "./output"


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":

    run_folder_inference(
        input_folder=INPUT_FOLDER,
        output_folder=OUTPUT_FOLDER,
        pipeline_fn=process_ocr_pipeline
    )