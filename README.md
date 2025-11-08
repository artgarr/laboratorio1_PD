# AutoML con DVC: Pipeline reproducible

## Requisitos  
Python 3.10 o superior, Git y DVC instalados.

## Experimentos realizados  
Se llevaron a cabo 4 experimentos utilizando distintas versiones del dataset original. Cada versión fue diseñada para evaluar la robustez del pipeline ante modificaciones en los datos:  
- **Versión 1**: dataset original sin modificaciones
- **Versión 2**: se eliminó una columna completa para analizar la sensibilidad del modelo ante pérdida de información estructural  
- **Versión 3**: se agregaron datos con ruido para simular variabilidad real  
- **Versión 4**: se eliminaron registros para evaluar el impacto de datos faltantes  

Cada versión fue integrada al pipeline mediante `params.yaml`, versionada con DVC y evaluada con métricas comparables usando `dvc metrics diff`. Esto permitió validar el comportamiento del modelo ante distintos escenarios de calidad y estructura de datos.

## Descarga y preparación  
Clona el repositorio con:  
`git clone https://github.com/artgarr/laboratorio1_PD.git`  

Navega a la carpeta del proyecto con `cd`, y asegúrate de tener el dataset correspondiente en la carpeta `data/` (por defecto se usa `dataset_v4.csv`). Si deseas usar otro dataset, cópialo a `data/` y luego ejecuta:  
`dvc add data/dataset_vx.csv`  
`git add data/dataset_vx.csv.dvc`  
`git commit -m "Agrega dataset vx"`  
Después, actualiza el archivo `params.yaml` con el nuevo nombre del dataset:  
`sed -i 's/dataset_v4.csv/dataset_vx.csv/' params.yaml`  
`git add params.yaml`  
`git commit -m "Actualiza a dataset vx"`

## Ejecución del pipeline  
Ejecuta el pipeline completo con:  
`dvc repro`  
Esto correrá las etapas `preprocess`, `train` y `evaluate`, que limpian los datos, entrenan los modelos definidos en `params.yaml` y generan las métricas en la carpeta local (`metrics.json`) y el reporte (`results.csv`).

## Visualización y comparación de métricas  
Para ver las métricas actuales, usa:  
`dvc metrics show`  
Para comparar métricas entre versiones, usa:  
`dvc metrics diff`  
o  
`dvc metrics diff --all`  
Esto permite evaluar el impacto de cambios en los datos o en la configuración del modelo de forma reproducible y trazable.
