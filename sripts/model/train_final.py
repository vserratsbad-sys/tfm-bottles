"""
Entrenamiento final con los mejores hiperparámetros encontrados por Optuna.
Lee best_trial.json y lanza un entrenamiento largo con el mismo data-aug
que usó Optuna. Al terminar:

  1. Carga best.pt (el mejor checkpoint del entrenamiento final).
  2. Evalúa por SEPARADO en  validación  y  test.
  3. Guarda TODAS las gráficas en disco (headless, nunca se muestran),
     organizadas en carpetas por tipo:

     runs/final/best_from_trial_XXX/
     ├── weights/                 ← best.pt, last.pt
     ├── results.csv              ← log de entrenamiento
     ├── eval_val/                ← salida nativa YOLO sobre validación (cruda)
     ├── eval_test/               ← salida nativa YOLO sobre test (cruda)
     └── plots_tfm/
         ├── entrenamiento/       ← curvas de loss/métricas + LR (PDF/SVG)
         ├── matrices_confusion/  ← confusión val + test (PNG nativo + vector)
         ├── curvas_pr/           ← PR / P / R / F1  val + test
         ├── metricas_por_clase/  ← mAP por clase  val + test (PDF/SVG)
         └── resumen/             ← val-vs-test, tabla hiperparámetros, CSV/JSON

Uso:
    python train_final.py
"""

import json
import shutil
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless: guarda a disco, nunca abre ventana
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────────────
#  CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────
# Todo cuelga de la carpeta del script, NO del cwd → no más "sitios raros".
DATA_YAML = str(BASE.parent.parent / "data.yaml")

BEST_JSON  = BASE / "runs/hpo/best_trial.json"
DATA_YAML  = str(BASE / "data.yaml")    # ⚠️ debe definir train/val/TEST
MODEL_BASE = "yolo26n.pt"
EPOCHS     = 300
IMGSZ      = 640
DEVICE     = 0          # 0 = GPU, "cpu" = sin GPU
WORKERS    = 4
# ─────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          10,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.25,
    "grid.linestyle":     "--",
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.15,
})

PALETTE = {
    "train":   "#2C6FBF",
    "val":     "#E05252",
    "test":    "#3BAF72",
    "map50":   "#9C27B0",
    "map5095": "#F4A21A",
    "prec":    "#3BAF72",
    "recall":  "#FF7043",
    "bg":      "#F7F8FA",
}

VECTOR_EXTS = ("pdf", "svg")


# ════════════════════════════════════════════════════════════════════
#  GRÁFICAS DE ENTRENAMIENTO (curvas a lo largo de épocas)
# ════════════════════════════════════════════════════════════════════
def plot_training(results_csv: Path, out_dir: Path, best: dict):
    if not results_csv.exists():
        print(f"  ⚠️  No se encuentra {results_csv}")
        return

    df = pd.read_csv(results_csv)
    df.columns = [c.strip() for c in df.columns]
    epochs = np.arange(1, len(df) + 1)

    def col(name):
        return df[name].values if name in df.columns else np.zeros(len(df))

    # ── FIGURA 1 · curvas de loss + métricas ──────────────────────────
    fig1 = plt.figure(figsize=(16, 10))
    fig1.patch.set_facecolor(PALETTE["bg"])
    fig1.suptitle(
        f"Entrenamiento final YOLO  —  mejor trial Optuna #{best['trial']}\n"
        f"mAP50-95(HPO)={best['map50_95']:.4f}  |  lr0={best['lr0']:.5f}"
        f"  |  opt={best['optimizer']}  |  batch={best['batch']}",
        fontsize=12, fontweight="bold"
    )
    gs = gridspec.GridSpec(2, 4, figure=fig1, hspace=0.42, wspace=0.35)

    def make_ax(pos, title, y_train, y_val=None,
                c_train=PALETTE["train"], c_val=PALETTE["val"]):
        ax = fig1.add_subplot(pos)
        ax.set_facecolor(PALETTE["bg"])
        ax.plot(epochs, y_train, color=c_train, lw=1.8, label="Train")
        if y_val is not None:
            ax.plot(epochs, y_val, color=c_val, lw=1.8, ls="--", label="Val")
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_xlabel("Época", fontsize=8)
        ax.legend(fontsize=7)
        return ax

    make_ax(gs[0, 0], "Box loss", col("train/box_loss"), col("val/box_loss"))
    make_ax(gs[0, 1], "Cls loss", col("train/cls_loss"), col("val/cls_loss"))
    make_ax(gs[0, 2], "DFL loss", col("train/dfl_loss"), col("val/dfl_loss"))

    ax_lr = fig1.add_subplot(gs[0, 3])
    ax_lr.set_facecolor(PALETTE["bg"])
    for lr_col in [c for c in df.columns if c.startswith("lr/")]:
        ax_lr.plot(epochs, df[lr_col].values, lw=1.5, label=lr_col.split("/")[-1])
    ax_lr.set_title("Learning rate", fontsize=9, fontweight="bold")
    ax_lr.set_xlabel("Época", fontsize=8)
    ax_lr.set_yscale("log")
    ax_lr.legend(fontsize=7)

    make_ax(gs[1, 0], "Precision", col("metrics/precision(B)"), c_train=PALETTE["prec"])
    make_ax(gs[1, 1], "Recall",    col("metrics/recall(B)"),    c_train=PALETTE["recall"])
    make_ax(gs[1, 2], "mAP@50",    col("metrics/mAP50(B)"),     c_train=PALETTE["map50"])
    make_ax(gs[1, 3], "mAP@50-95", col("metrics/mAP50-95(B)"),  c_train=PALETTE["map5095"])

    for ext in VECTOR_EXTS:
        fig1.savefig(out_dir / f"curvas_entrenamiento.{ext}", format=ext)
    plt.close(fig1)

    # ── FIGURA 2 · convergencia mAP50-95 (mejor época marcada) ────────
    m = col("metrics/mAP50-95(B)")
    if m.max() > 0:
        best_epoch = int(m.argmax()) + 1
        fig2, ax = plt.subplots(figsize=(9, 5))
        fig2.patch.set_facecolor(PALETTE["bg"])
        ax.set_facecolor(PALETTE["bg"])
        ax.plot(epochs, m, color=PALETTE["map5095"], lw=2, label="mAP50-95 (val)")
        ax.axvline(best_epoch, color="red", lw=1.2, ls=":", label=f"Mejor época #{best_epoch}")
        ax.scatter(best_epoch, m[best_epoch - 1], s=90, color="red", zorder=6)
        ax.set_xlabel("Época"); ax.set_ylabel("mAP50-95")
        ax.set_title("Convergencia mAP50-95 durante el entrenamiento")
        ax.legend(fontsize=8)
        fig2.tight_layout()
        for ext in VECTOR_EXTS:
            fig2.savefig(out_dir / f"convergencia_map.{ext}", format=ext)
        plt.close(fig2)

    print(f"  ✅  entrenamiento → {out_dir.name}/")


# ════════════════════════════════════════════════════════════════════
#  EVALUACIÓN POR SUBCONJUNTO (val / test)
# ════════════════════════════════════════════════════════════════════
def global_metrics(m) -> dict:
    b = m.box
    return {
        "mAP50-95":  float(b.map),
        "mAP50":     float(b.map50),
        "Precision": float(b.mp),
        "Recall":    float(b.mr),
    }


def per_class_rows(m, names) -> list:
    """[(clase, mAP50-95, mAP50, P, R), ...] solo para clases presentes."""
    rows = []
    try:
        b = m.box
        idx = list(b.ap_class_index)
        for i, c in enumerate(idx):
            name = names.get(c, str(c)) if isinstance(names, dict) else names[c]
            map5095 = float(b.ap[i].mean()) if hasattr(b, "ap") else float(b.maps[c])
            map50   = float(b.ap50[i])      if hasattr(b, "ap50") else 0.0
            p       = float(b.p[i])         if hasattr(b, "p") and len(b.p) > i else 0.0
            r       = float(b.r[i])         if hasattr(b, "r") and len(b.r) > i else 0.0
            rows.append((name, map5095, map50, p, r))
    except Exception as e:
        print(f"  (métricas por clase no disponibles: {e})")
    return rows


def collect_native_plots(save_dir: Path, split: str, dir_cm: Path, dir_pr: Path):
    """Copia las gráficas nativas de YOLO a las carpetas por tipo, con sufijo de split."""
    patterns = [
        ("confusion_matrix*.png", dir_cm),
        ("*PR_curve*.png",        dir_pr),
        ("*P_curve*.png",         dir_pr),
        ("*R_curve*.png",         dir_pr),
        ("*F1_curve*.png",        dir_pr),
    ]
    for pat, dest in patterns:
        for src in save_dir.glob(pat):
            shutil.copy(src, dest / f"{src.stem}_{split}{src.suffix}")


def plot_per_class(rows, split, out_dir):
    if not rows:
        return
    rows = sorted(rows, key=lambda x: x[1])
    names = [r[0] for r in rows]
    vals  = [r[1] for r in rows]
    color = PALETTE["val"] if split == "val" else PALETTE["test"]
    fig, ax = plt.subplots(figsize=(8, max(3, len(rows) * 0.42)))
    ax.barh(names, vals, color=color, alpha=0.85)
    ax.set_xlim(0, 1); ax.set_xlabel("mAP@50-95")
    ax.set_title(f"mAP50-95 por clase — {split}")
    for i, v in enumerate(vals):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    for ext in VECTOR_EXTS:
        fig.savefig(out_dir / f"map_por_clase_{split}.{ext}", format=ext)
    plt.close(fig)


def plot_confusion_vector(m, names, split, out_dir):
    """Matriz de confusión en formato vectorial (si la API lo permite)."""
    try:
        cm = getattr(m, "confusion_matrix", None)
        if cm is None or not hasattr(cm, "matrix"):
            return
        matrix = np.asarray(cm.matrix, dtype=float)
        n = matrix.shape[0]
        labels = [names[i] if i in names else str(i) for i in range(n - 1)] + ["fondo"]
        with np.errstate(all="ignore"):
            norm = np.nan_to_num(matrix / matrix.sum(axis=0, keepdims=True))

        fig, ax = plt.subplots(figsize=(max(6, n * 0.6), max(5, n * 0.55)))
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Verdadero"); ax.set_ylabel("Predicho")
        ax.set_title(f"Matriz de confusión (norm. por columna) — {split}")
        if n <= 12:    # solo anotar si no satura
            for i in range(n):
                for j in range(n):
                    if norm[i, j] > 0.005:
                        ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center",
                                fontsize=7, color="white" if norm[i, j] > 0.5 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        for ext in VECTOR_EXTS:
            fig.savefig(out_dir / f"matriz_confusion_{split}.{ext}", format=ext)
        plt.close(fig)
    except Exception as e:
        print(f"  (matriz vectorial {split} no disponible, queda el PNG nativo: {e})")


def evaluate_split(model, split, run_dir, dirs):
    """Evalúa best.pt en un split, copia gráficas nativas y genera las vectoriales."""
    print(f"\n  ▶ Evaluando split = {split} …")
    try:
        m = model.val(
            data=DATA_YAML, split=split, imgsz=IMGSZ, device=DEVICE,
            project=str(run_dir), name=f"eval_{split}", exist_ok=True,
            plots=True, verbose=False,
        )
    except Exception as e:
        print(f"  ❌ No se pudo evaluar split={split}: {e}")
        print(f"     (¿está '{split}' definido en data.yaml?)")
        return None

    save_dir = run_dir / f"eval_{split}"
    collect_native_plots(save_dir, split, dirs["cm"], dirs["pr"])

    names = model.names if hasattr(model, "names") else {}
    plot_confusion_vector(m, names, split, dirs["cm"])
    plot_per_class(per_class_rows(m, names), split, dirs["per_class"])

    g = global_metrics(m)
    print(f"    {split}: mAP50-95={g['mAP50-95']:.4f}  mAP50={g['mAP50']:.4f}"
          f"  P={g['Precision']:.4f}  R={g['Recall']:.4f}")
    return g


# ════════════════════════════════════════════════════════════════════
#  RESUMEN val vs test  +  tabla de hiperparámetros
# ════════════════════════════════════════════════════════════════════
def plot_summary(metrics_by_split: dict, best: dict, out_dir: Path):
    keys = ["mAP50-95", "mAP50", "Precision", "Recall"]
    splits = [s for s in ("val", "test") if metrics_by_split.get(s)]

    # ── Comparativa val vs test ──
    if splits:
        x = np.arange(len(keys)); w = 0.8 / len(splits)
        fig, ax = plt.subplots(figsize=(10, 5.5))
        fig.patch.set_facecolor(PALETTE["bg"]); ax.set_facecolor(PALETTE["bg"])
        for i, s in enumerate(splits):
            vals = [metrics_by_split[s][k] for k in keys]
            bars = ax.bar(x + i * w - (len(splits) - 1) * w / 2, vals, w,
                          label=s, color=PALETTE[s], alpha=0.85)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                        ha="center", fontsize=8, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(keys)
        ax.set_ylim(0, 1.08); ax.set_ylabel("Valor")
        ax.set_title("Rendimiento del modelo final — validación vs test")
        ax.legend()
        fig.tight_layout()
        for ext in VECTOR_EXTS:
            fig.savefig(out_dir / f"val_vs_test.{ext}", format=ext)
        plt.close(fig)

        # CSV + JSON con los números (para Tabla del TFM)
        pd.DataFrame(metrics_by_split).T.to_csv(out_dir / "metricas_resumen.csv")
        (out_dir / "metricas_resumen.json").write_text(
            json.dumps(metrics_by_split, indent=2, ensure_ascii=False))

    # ── Tabla de hiperparámetros usados ──
    hp = [
        ("Optimizer", best["optimizer"]), ("lr0", f"{best['lr0']:.6f}"),
        ("lrf", f"{best['lrf']:.4f}"), ("Momentum", f"{best['momentum']:.4f}"),
        ("Weight decay", f"{best['weight_decay']:.2e}"),
        ("Warmup epochs", str(best["warmup_epochs"])),
        ("Box gain", f"{best['box']:.4f}"), ("Cls gain", f"{best['cls']:.4f}"),
        ("DFL gain", f"{best['dfl']:.4f}"), ("Batch", str(best["batch"])),
        ("hsv_h", f"{best['hsv_h']:.4f}"), ("hsv_s", f"{best['hsv_s']:.4f}"),
        ("hsv_v", f"{best['hsv_v']:.4f}"), ("Translate", f"{best['translate']:.4f}"),
        ("Scale", f"{best['scale']:.4f}"), ("Mosaic", f"{best['mosaic']:.4f}"),
        ("Trial origen", str(best["trial"])), ("mAP50-95 HPO", f"{best['map50_95']:.4f}"),
    ]
    fig, ax = plt.subplots(figsize=(7, len(hp) * 0.42 + 1.5))
    fig.patch.set_facecolor(PALETTE["bg"]); ax.set_facecolor(PALETTE["bg"]); ax.axis("off")
    ax.set_title("Hiperparámetros del entrenamiento final", fontsize=12, fontweight="bold", pad=12)
    table = ax.table(cellText=hp, colLabels=["Parámetro", "Valor"], loc="center", cellLoc="left")
    table.auto_set_font_size(False); table.set_fontsize(9.5); table.scale(1, 1.35)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#CCCCCC")
        if row == 0:
            cell.set_facecolor("#2C6FBF"); cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#E8EEF6" if row % 2 == 0 else "#F7F8FA")
    fig.tight_layout()
    for ext in VECTOR_EXTS:
        fig.savefig(out_dir / f"hiperparametros.{ext}", format=ext)
    plt.close(fig)
    print(f"  ✅  resumen → {out_dir.name}/")


# ════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════
def main():
    if not BEST_JSON.exists():
        raise FileNotFoundError(f"No se encuentra {BEST_JSON.resolve()}")
    best = json.loads(BEST_JSON.read_text())

    print("╔══════════════════════════════════════════════╗")
    print("║   Entrenamiento final — mejores parámetros     ║")
    print("╚══════════════════════════════════════════════╝")
    print(f"  Trial origen : #{best['trial']}")
    print(f"  mAP50-95 HPO : {best['map50_95']:.4f}")
    print(f"  Epochs       : {EPOCHS}")
    print(f"  Data         : {DATA_YAML}")

    # ── ENTRENAMIENTO ────────────────────────────────────────────────
    model = YOLO(MODEL_BASE)
    model.train(
        data=DATA_YAML, epochs=EPOCHS, imgsz=IMGSZ, device=DEVICE, workers=WORKERS,
        project=str(BASE / "runs/final"),
        name=f"best_from_trial_{best['trial']:03d}",
        exist_ok=True, verbose=True,
        optimizer=best["optimizer"], lr0=best["lr0"], lrf=best["lrf"],
        momentum=best["momentum"], weight_decay=best["weight_decay"],
        warmup_epochs=best["warmup_epochs"],
        box=best["box"], cls=best["cls"], dfl=best["dfl"],
        # Augmentation online de YOLO con los valores hallados por Optuna
        hsv_h=best["hsv_h"], hsv_s=best["hsv_s"], hsv_v=best["hsv_v"],
        translate=best["translate"], scale=best["scale"], mosaic=best["mosaic"],
        # Solo se desactivan rotaciones y volteos (sin sentido para botellas)
        degrees=0.0, flipud=0.0, fliplr=0.0,
        batch=best["batch"],
    )
    run_dir = Path(model.trainer.save_dir)
    print(f"\n  📁 Run dir: {run_dir}")

    # ── Carpetas de salida organizadas ───────────────────────────────
    plots = run_dir / "plots_tfm"
    dirs = {
        "entreno":   plots / "entrenamiento",
        "cm":        plots / "matrices_confusion",
        "pr":        plots / "curvas_pr",
        "per_class": plots / "metricas_por_clase",
        "resumen":   plots / "resumen",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # ── Gráficas de entrenamiento ────────────────────────────────────
    print("\n  📊 Gráficas de entrenamiento …")
    plot_training(run_dir / "results.csv", dirs["entreno"], best)

    # ── Evaluación best.pt en val y test ─────────────────────────────
    best_pt = run_dir / "weights" / "best.pt"
    eval_model = YOLO(str(best_pt)) if best_pt.exists() else model
    if not best_pt.exists():
        print("  ⚠️  No se encontró best.pt; evalúo con el modelo en memoria.")

    metrics_by_split = {}
    for split in ("val", "test"):
        g = evaluate_split(eval_model, split, run_dir, dirs)
        if g:
            metrics_by_split[split] = g

    # ── Resumen final ────────────────────────────────────────────────
    print("\n  📊 Resumen val vs test + hiperparámetros …")
    plot_summary(metrics_by_split, best, dirs["resumen"])

    print("\n✅ Entrenamiento final completado.")
    print(f"   Modelo    → {best_pt}")
    print(f"   Gráficas  → {plots}/")
    for name, d in dirs.items():
        print(f"     · {d.relative_to(run_dir)}/")
    if metrics_by_split:
        print("\n   RESUMEN:")
        for s, g in metrics_by_split.items():
            print(f"     {s:<5} mAP50-95={g['mAP50-95']:.4f}  mAP50={g['mAP50']:.4f}"
                  f"  P={g['Precision']:.4f}  R={g['Recall']:.4f}")


if __name__ == "__main__":
    main()