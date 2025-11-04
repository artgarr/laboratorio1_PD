import pandas as pd
import yaml

# Cargar parámetros
params = yaml.safe_load(open("params.yaml"))
raw_path = params["data"]["raw"]
processed_path = params["data"]["processed"]

# Leer datos
df = pd.read_csv(raw_path)

# Ejemplo simple: eliminar duplicados
df = df.drop_duplicates()

# Guardar procesado
df.to_csv(processed_path, index=False)
print(f"Datos procesados en {processed_path}")