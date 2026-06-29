"""
╔══════════════════════════════════════════════════════════════════════╗
║         YOLO26n · Optuna HPO · Sistema de registro completo         ║
║                   FIX: métricas desde results object               ║
╚══════════════════════════════════════════════════════════════════════╝

Genera por cada trial:
  runs/hpo/
  ├── trial_000/          ← carpeta de entrenamiento YOLO
  ├── trial_001/
  ├── ...
  ├── plots/              ← gráficas PNG de cada trial + comparativas
  ├── log.csv             ← registro completo de todos los trials
  ├── log.json            ← ídem en JSON
  ├── best_trial.json     ← info del mejor trial
  └── optuna_study.db     ← base de datos Optuna (reanudable)

Uso:
    pip install ultralytics optuna matplotlib pandas
    python train_optuna.py

Test rápido (2 trials, 3 épocas):
    QUICK_TEST=1 python train_optuna.py
"""

import json
import csv
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

import optuna
import matplotlib
matplotlib.use("Agg")   # sin GUI
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────────────
#  MODO TEST RÁPIDO  (QUICK_TEST=1 python train_optuna.py)
# ─────────────────────────────────────────────────────────────────────
QUICK_TEST = os.environ.get("QUICK_TEST", "0") == "1"

# ─────────────────────────────────────────────────────────────────────
#  CONFIGURACIÓN  ←  edita aquí
# ─────────────────────────────────────────────────────────────────────
from pathlib import Path

# Carpeta donde está ESTE script → todo cuelga de aquí, no del cwd
DATA_YAML = str(BASE.parent.parent / "data.yaml")

DATA_YAML    = str(BASE / "data.yaml")   # ⚠️ si tu data.yaml está en otra carpeta, pon la ruta aquí
MODEL_BASE   = "yolo26n.pt"
N_TRIALS     = 2  if QUICK_TEST else 30
EPOCHS       = 3  if QUICK_TEST else 50
IMGSZ        = 640
DEVICE       = 0
WORKERS      = 4
METRIC       = "metrics/mAP50-95(B)"
HPO_DIR      = BASE / ("runs/hpo_test" if QUICK_TEST else "runs/hpo")
# ─────────────────────────────────────────────────────────────────────

HPO_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR = HPO_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

LOG_CSV   = HPO_DIR / "log.csv"
LOG_JSON  = HPO_DIR / "log.json"
BEST_JSON = HPO_DIR / "best_trial.json"
DB_PATH   = Path("/tmp") / ("optuna_study_test.db" if QUICK_TEST else "optuna_study.db")

CSV_FIELDS = [
    "trial", "timestamp", "status",
    "lr0", "lrf", "momentum", "weight_decay",
    "warmup_epochs", "box", "cls", "dfl",
    "hsv_h", "hsv_s", "hsv_v",
    "translate", "scale", "mosaic",
    "batch", "optimizer",
    "map50", "map50_95", "precision", "recall",
    "best_epoch", "duration_min", "is_best", "is_last",
]

# ── Inicializar CSV ──────────────────────────────────────────────────
if not LOG_CSV.exists():
    with open(LOG_CSV, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()


# ════════════════════════════════════════════════════════════════════
#  ESPACIO DE BÚSQUEDA
# ════════════════════════════════════════════════════════════════════
def sample_params(trial: optuna.Trial) -> dict:
    return {
        "optimizer":     trial.suggest_categorical("optimizer", ["SGD", "Adam", "AdamW"]),
        "lr0":           trial.suggest_float("lr0", 1e-4, 1e-1, log=True),
        "lrf":           trial.suggest_float("lrf", 0.01, 0.3),
        "momentum":      trial.suggest_float("momentum", 0.80, 0.98),
        "weight_decay":  trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
        "warmup_epochs": trial.suggest_int("warmup_epochs", 1, 5),
        "box":           trial.suggest_float("box", 3.0, 10.0),
        "cls":           trial.suggest_float("cls", 0.2, 2.0),
        "dfl":           trial.suggest_float("dfl", 0.5, 2.0),
        "hsv_h":         trial.suggest_float("hsv_h", 0.0, 0.1),
        "hsv_s":         trial.suggest_float("hsv_s", 0.0, 0.9),
        "hsv_v":         trial.suggest_float("hsv_v", 0.0, 0.5),
        "translate":     trial.suggest_float("translate", 0.0, 0.2),
        "scale":         trial.suggest_float("scale", 0.2, 0.9),
        "mosaic":        trial.suggest_float("mosaic", 0.5, 1.0),
        "batch":         trial.suggest_categorical("batch", [8, 16, 32]),
    }


# ════════════════════════════════════════════════════════════════════
#  GRÁFICAS POR TRIAL
# ════════════════════════════════════════════════════════════════════
def find_trial_dir(trial_id: int) -> Path | None:
    """
    Busca la carpeta real del trial aunque YOLO haya añadido sufijo
    (trial_000, trial_0002, trial_0003...).
    Devuelve el Path más reciente que coincida.
    """
    candidates = sorted(HPO_DIR.glob(f"trial_{trial_id:03d}*/"))
    return candidates[-1] if candidates else None


def plot_trial(results_csv: Path, trial_id: int, params: dict, metrics: dict):
    """Genera PNG con las curvas del trial."""
    try:
        df = pd.read_csv(results_csv)
        df.columns = [c.strip() for c in df.columns]
    except Exception:
        return

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(
        f"Trial {trial_id:03d}  |  mAP50-95={metrics.get('map50_95', 0):.4f}"
        f"  |  lr0={params['lr0']:.5f}  opt={params['optimizer']}",
        fontsize=13, fontweight="bold"
    )

    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.4, wspace=0.35)

    def safe_plot(ax, col, title, color="steelblue"):
        if col in df.columns:
            ax.plot(df[col], color=color, linewidth=1.5)
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("epoch", fontsize=8)
            ax.grid(True, alpha=0.3)
        else:
            ax.set_visible(False)

    safe_plot(fig.add_subplot(gs[0, 0]), "train/box_loss",  "Train Box Loss",  "#e07b54")
    safe_plot(fig.add_subplot(gs[0, 1]), "train/cls_loss",  "Train Cls Loss",  "#e07b54")
    safe_plot(fig.add_subplot(gs[0, 2]), "val/box_loss",    "Val Box Loss",    "#5488c8")
    safe_plot(fig.add_subplot(gs[0, 3]), "val/cls_loss",    "Val Cls Loss",    "#5488c8")

    safe_plot(fig.add_subplot(gs[1, 0]), "metrics/precision(B)", "Precision",  "#4caf50")
    safe_plot(fig.add_subplot(gs[1, 1]), "metrics/recall(B)",    "Recall",     "#ff9800")
    safe_plot(fig.add_subplot(gs[1, 2]), "metrics/mAP50(B)",     "mAP@50",     "#9c27b0")
    safe_plot(fig.add_subplot(gs[1, 3]), "metrics/mAP50-95(B)",  "mAP@50-95",  "#f44336")

    out = PLOTS_DIR / f"trial_{trial_id:03d}.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 Gráfica guardada → {out}")


def plot_summary(log: list[dict]):
    """Gráfica comparativa de todos los trials hasta el momento."""
    if len(log) < 2:
        return
    ok_trials = [r for r in log if r["status"] == "ok"]
    if not ok_trials:
        return
    trials  = [r["trial"]    for r in ok_trials]
    map5095 = [r["map50_95"] for r in ok_trials]
    map50   = [r["map50"]    for r in ok_trials]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Resumen HPO — todos los trials", fontsize=13, fontweight="bold")

    best_idx = map5095.index(max(map5095))

    for ax, vals, title, color in [
        (axes[0], map5095, "mAP50-95", "#f44336"),
        (axes[1], map50,   "mAP50",    "#9c27b0"),
    ]:
        ax.bar(trials, vals, color=color, alpha=0.6)
        ax.bar(trials[best_idx], vals[best_idx], color="gold", label="Mejor")
        ax.set_xlabel("Trial")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)

    fig.savefig(PLOTS_DIR / "summary.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════
#  REGISTRO
# ════════════════════════════════════════════════════════════════════
def load_log() -> list[dict]:
    if LOG_JSON.exists():
        return json.loads(LOG_JSON.read_text())
    return []


def save_log(log: list[dict]):
    LOG_JSON.write_text(json.dumps(log, indent=2, ensure_ascii=False))


def append_csv(row: dict):
    with open(LOG_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writerow(row)


def update_best_flags(log: list[dict]) -> list[dict]:
    ok = [r for r in log if r["status"] == "ok"]
    if not ok:
        return log
    best_val = max(r["map50_95"] for r in ok)
    best_id  = next(r["trial"] for r in ok if r["map50_95"] == best_val)
    last_id  = max(r["trial"] for r in ok)
    for r in log:
        r["is_best"] = (r["trial"] == best_id and r["status"] == "ok")
        r["is_last"] = (r["trial"] == last_id and r["status"] == "ok")
    return log


# ════════════════════════════════════════════════════════════════════
#  OBJETIVO OPTUNA
# ════════════════════════════════════════════════════════════════════
def objective(trial: optuna.Trial) -> float:
    trial_id  = trial.number
    params    = sample_params(trial)
    t0        = time.time()
    timestamp = datetime.now().isoformat(timespec="seconds")

    print(f"\n{'═'*60}")
    print(f"  Trial {trial_id:03d}  —  {timestamp}")
    print(f"  Params: {params}")
    print(f"{'═'*60}")

    row = {
        "trial": trial_id, "timestamp": timestamp, "status": "error",
        "map50": 0, "map50_95": 0, "precision": 0, "recall": 0,
        "best_epoch": 0, "duration_min": 0, "is_best": False, "is_last": False,
        **params
    }

    try:
        model = YOLO(MODEL_BASE)
        results = model.train(
            data          = DATA_YAML,
            epochs        = EPOCHS,
            imgsz         = IMGSZ,
            device        = DEVICE,
            workers       = WORKERS,
            project       = str(HPO_DIR),
            name          = f"trial_{trial_id:03d}",
            exist_ok      = True,
            verbose       = False,
            optimizer     = params["optimizer"],
            lr0           = params["lr0"],
            lrf           = params["lrf"],
            momentum      = params["momentum"],
            weight_decay  = params["weight_decay"],
            warmup_epochs = params["warmup_epochs"],
            box           = params["box"],
            cls           = params["cls"],
            dfl           = params["dfl"],
            hsv_h         = params["hsv_h"],
            hsv_s         = params["hsv_s"],
            hsv_v         = params["hsv_v"],
            translate     = params["translate"],
            scale         = params["scale"],
            mosaic        = params["mosaic"],
            batch         = params["batch"],
            degrees       = 0.0,
            flipud        = 0.0,
            fliplr        = 0.0,
        )

        # ── ✅ FIX PRINCIPAL: métricas desde el objeto results ────────
        # Esto funciona SIEMPRE, independientemente de la ruta que YOLO use
        rd = results.results_dict if hasattr(results, "results_dict") else {}

        row["map50_95"]  = float(rd.get("metrics/mAP50-95(B)", 0) or 0)
        row["map50"]     = float(rd.get("metrics/mAP50(B)",    0) or 0)
        row["precision"] = float(rd.get("metrics/precision(B)", 0) or 0)
        row["recall"]    = float(rd.get("metrics/recall(B)",    0) or 0)
        metric_val       = row["map50_95"]

        # ── Fallback: leer desde results.csv si results_dict viene vacío ──
        if metric_val == 0:
            trial_dir = find_trial_dir(trial_id)
            if trial_dir:
                res_csv = trial_dir / "results.csv"
                if res_csv.exists():
                    df = pd.read_csv(res_csv)
                    df.columns = [c.strip() for c in df.columns]
                    col_map = {
                        "map50_95":  "metrics/mAP50-95(B)",
                        "map50":     "metrics/mAP50(B)",
                        "precision": "metrics/precision(B)",
                        "recall":    "metrics/recall(B)",
                    }
                    for key, col in col_map.items():
                        if col in df.columns:
                            row[key] = float(df[col].max())
                    metric_val = row["map50_95"]
                    if "metrics/mAP50-95(B)" in df.columns:
                        row["best_epoch"] = int(df["metrics/mAP50-95(B)"].idxmax()) + 1
                    print(f"  ⚠️  results_dict vacío, leído desde {res_csv}")

        # best_epoch desde results_dict si está disponible
        if hasattr(results, "best_epoch") and results.best_epoch is not None:
            row["best_epoch"] = int(results.best_epoch)

        row["status"]       = "ok"
        row["duration_min"] = round((time.time() - t0) / 60, 2)

        # ── Gráfica del trial ─────────────────────────────────────
        trial_dir = find_trial_dir(trial_id)
        if trial_dir:
            res_csv = trial_dir / "results.csv"
            if res_csv.exists():
                plot_trial(res_csv, trial_id, params, row)
            for src_name in ["confusion_matrix.png", "PR_curve.png", "F1_curve.png"]:
                src = trial_dir / src_name
                if src.exists():
                    shutil.copy(src, PLOTS_DIR / f"trial_{trial_id:03d}_{src_name}")

        print(f"  ✅ mAP50-95={row['map50_95']:.4f}  mAP50={row['map50']:.4f}"
              f"  precision={row['precision']:.4f}  recall={row['recall']:.4f}"
              f"  [{row['duration_min']} min]")
        return metric_val

    except Exception as e:
        import traceback
        print(f"  ❌ Trial {trial_id} falló: {e}")
        traceback.print_exc()
        row["status"] = f"error: {e}"
        return 0.0

    finally:
        log = load_log()
        log = [r for r in log if r["trial"] != trial_id]
        log.append(row)
        log = update_best_flags(log)
        save_log(log)
        append_csv(row)

        best = next((r for r in log if r.get("is_best")), None)
        if best:
            BEST_JSON.write_text(json.dumps(best, indent=2, ensure_ascii=False))
            print(f"  🏆 Mejor hasta ahora: trial {best['trial']}"
                  f"  mAP50-95={best['map50_95']:.4f}")

        plot_summary(log)


# ════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════
def main():
    mode = "🧪 QUICK TEST (2 trials, 3 épocas)" if QUICK_TEST else "🚀 PRODUCCIÓN"
    print("╔══════════════════════════════════════════════╗")
    print(f"║   YOLO26n · Optuna HPO  [{mode}]")
    print(f"║   Trials: {N_TRIALS:<5}  Epochs/trial: {EPOCHS:<5}      ║")
    print(f"║   Output: {HPO_DIR}                         ║")
    print("╚══════════════════════════════════════════════╝\n")

    storage = f"sqlite:///{DB_PATH}"
    study = optuna.create_study(
        study_name     = "yolo26n_hpo_test" if QUICK_TEST else "yolo26n_hpo",
        direction      = "maximize",
        storage        = storage,
        load_if_exists = True,
        sampler        = optuna.samplers.TPESampler(seed=42),
        pruner         = optuna.pruners.MedianPruner(n_startup_trials=5,
                                                      n_warmup_steps=10),
    )

    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    # ── Resumen final ─────────────────────────────────────────────
    best = study.best_trial
    print("\n" + "═" * 60)
    print(f"  🏆 MEJOR TRIAL: #{best.number}")
    print(f"  mAP50-95: {best.value:.4f}")
    print(f"  Parámetros:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")
    print("═" * 60)

    # ── Gráficas Optuna nativas ───────────────────────────────────
    try:
        from optuna.visualization.matplotlib import (
            plot_optimization_history,
            plot_param_importances,
            plot_parallel_coordinate,
        )
        fig, axes = plt.subplots(1, 3, figsize=(22, 6))
        fig.suptitle("Optuna — Análisis HPO", fontsize=14, fontweight="bold")

        plot_optimization_history(study, ax=axes[0])
        axes[0].set_title("Historial de optimización")

        plot_param_importances(study, ax=axes[1])
        axes[1].set_title("Importancia de parámetros")

        plot_parallel_coordinate(study, ax=axes[2])
        axes[2].set_title("Coordenadas paralelas")

        fig.savefig(PLOTS_DIR / "optuna_analysis.png", dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  📊 Análisis Optuna → {PLOTS_DIR / 'optuna_analysis.png'}")
    except Exception as e:
        print(f"  (Gráficas Optuna no disponibles: {e})")

    print(f"\n  📁 Log CSV  → {LOG_CSV}")
    print(f"  📁 Log JSON → {LOG_JSON}")
    print(f"  📁 Mejor   → {BEST_JSON}")
    print(f"  📁 Plots   → {PLOTS_DIR}/\n")

    # ── Verificación rápida post-ejecución ────────────────────────
    if LOG_CSV.exists():
        df = pd.read_csv(LOG_CSV)
        print("  📋 RESUMEN LOG:")
        print(df[["trial", "status", "map50", "map50_95", "precision", "recall",
                   "optimizer", "batch", "duration_min"]].to_string(index=False))
        zeros = (df["map50_95"] == 0).sum()
        if zeros == len(df):
            print("\n  ⚠️  AVISO: Todos los mAP son 0. Comprueba data.yaml y el dataset.")
        elif zeros > 0:
            print(f"\n  ⚠️  {zeros} trials con mAP=0. Revisa esos trials.")
        else:
            print("\n  ✅ Todas las métricas se han registrado correctamente.")


if __name__ == "__main__":
    main()