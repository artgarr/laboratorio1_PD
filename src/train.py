import pandas as pd
import yaml, json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import numpy as np
from pathlib import Path

# Cargar parámetros desde YAML
config = yaml.safe_load(open("params.yaml"))
processed_file = config["data"]["processed"]
target_col = config["preprocess"]["target_column"]

# Leer dataset procesado
dataset = pd.read_csv(processed_file)

features = dataset.drop(target_col, axis=1)
labels = dataset[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    features, labels,
    test_size=config["preprocess"]["test_size"],
    random_state=config["preprocess"]["random_state"]
)

metrics_summary = {}
top_model = None
lowest_rmse = float("inf")
top_model_name = ""

# Entrenar modelos definidos en params.yaml
for model_cfg in config["automl"]["models"]:
    model_name = model_cfg["name"]

    if model_name == "linear":
        model = LinearRegression(**model_cfg.get("params", {}))
    elif model_name == "random_forest":
        model = RandomForestRegressor(**model_cfg.get("params", {}))
    elif model_name == "gradient_boosting":
        model = GradientBoostingRegressor(**model_cfg.get("params", {}))
    else:
        continue

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    rmse_val = float(np.sqrt(mean_squared_error(y_test, predictions)))
    r2_val = float(r2_score(y_test, predictions))
    mse_val = float(mean_squared_error(y_test, predictions))

    metrics_summary[model_name] = {"RMSE": rmse_val, "R2": r2_val, "MSE": mse_val}

    if rmse_val < lowest_rmse:
        lowest_rmse = rmse_val
        top_model = model
        top_model_name = model_name

output_dir = Path(config['artifacts']['dir'])
output_dir.mkdir(parents=True, exist_ok=True)

# Guardar mejor modelo y métricas
joblib.dump(top_model, output_dir / "best_model.pkl")
json.dump(metrics_summary, open("metrics.json", "w"), indent=4)

print(f"Modelo ganador: {top_model_name}")
print(f"RMSE={lowest_rmse:.4f}, R2={metrics_summary[top_model_name]['R2']:.4f}, MSE={metrics_summary[top_model_name]['MSE']:.4f}")

