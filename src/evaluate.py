import json
import pandas as pd

# Cargar métricas
metrics = json.load(open("metrics.json"))

# Convertir a DataFrame y guardar reporte
df = pd.DataFrame(metrics).T
df.to_csv("report.csv")

print("Reporte de métricas guardado en report.csv")
