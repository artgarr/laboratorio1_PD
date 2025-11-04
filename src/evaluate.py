import argparse
from pathlib import Path
import joblib
import csv
import json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--metrics", required=True)
    args = ap.parse_args()

    # Cargar resumen de entrenamiento
    summary_path = Path(args.models_dir) / "runs_summary.joblib"
    summary = joblib.load(summary_path)

    # Extraer métricas del mejor modelo
    metrics = {
        "best_model": summary["best"]["name"],
        "best_rmse": summary["best"]["metrics"]["rmse"],
        "best_mae": summary["best"]["metrics"]["mae"],
        "best_r2": summary["best"]["metrics"]["r2"],
    }

    # Guardar métricas
    Path(args.metrics).parent.mkdir(parents=True, exist_ok=True)
    with open(args.metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Guardar reporte tabular de todos los modelos
    rows = []
    for r in summary["runs"]:
        rows.append({
            "model": r["name"],
            "cv_rmse": r["cv_metrics"]["rmse"],
            "cv_mae": r["cv_metrics"]["mae"],
            "cv_r2": r["cv_metrics"]["r2"],
            "test_rmse": r["test_metrics"]["rmse"],
            "test_mae": r["test_metrics"]["mae"],
            "test_r2": r["test_metrics"]["r2"],
            "model_path": r["model_path"]
        })

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("Evaluation done:", args.report, args.metrics)

if __name__ == "__main__":
    main()
