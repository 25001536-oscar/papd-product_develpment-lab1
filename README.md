# AutoML con DVC — Laboratorio 1

Este repositorio contiene un pipeline reproducible de tipo AutoML gestionado con **DVC** y **Git**.  
Sigue una secuencia de tres etapas principales:

1. **preprocess** — limpia y escala las variables numéricas y guarda un archivo parquet preprocesado.  
2. **train** — ejecuta una *grid search* entre varios modelos y guarda el mejor.  
3. **evaluate** — evalúa el mejor modelo, genera `artifacts/metrics.json` y un breve informe en `reports/report.md`.

> Dataset utilizado por defecto: `data/dataset_v1.csv` (California Housing), columna objetivo `MedHouseVal`.

---

## 🚀 Inicio rápido

```bash
# 1) Clonar o crear el repositorio
git init
python -m venv .venv && source .venv/bin/activate  # (En Windows: .venv\Scripts\activate)
pip install -r requirements.txt

# 2) Inicializar DVC
dvc init

# 3) Agregar el dataset inicial a DVC (no subir archivos grandes a Git)
dvc add data/dataset_v1.csv
git add data/dataset_v1.csv.dvc .gitignore dvc.yaml params.yaml src requirements.txt
git commit -m "Inicio: pipeline DVC + dataset v1"

# 4) Ejecutar el pipeline completo
dvc repro

# 5) Mostrar las métricas obtenidas
dvc metrics show
```

---

## 🧩 Uso de nuevas versiones de dataset (dataset_v2.csv)

Se pueden crear y usar nuevas versiones del dataset para comparar resultados del modelo.  
Por ejemplo, el archivo **`data/dataset_v2.csv`** incluye limpieza ligera (sin duplicados, imputers de medianas, clipping de outliers, etc.).

### ➕ Agregar el nuevo dataset a DVC

```bash
dvc add data/dataset_v2.csv
git add data/dataset_v2.csv.dvc
git commit -m "Dataset v2: limpieza ligera"
```

### ⚙️ Actualizar el pipeline para usar dataset_v2

Editar el archivo `params.yaml` y reemplaza la ruta del dataset:

```yaml
dataset:
  path: data/dataset_v2.csv
  target: MedHouseVal
```

Guardar el archivo y vuelve a ejecutar el pipeline:

```bash
dvc repro
```

### 📊 Comparar métricas entre versiones

Se puede comparar comparar el rendimiento entre `dataset_v1` y `dataset_v2`:

```bash
# Comparar resultados con el commit anterior
dvc metrics diff HEAD~1

# O si se usaran etiquetas
git tag v1_dataset
# ... luego de ejecutar con v2 ...
git tag v2_dataset
dvc metrics diff v1_dataset v2_dataset
```

> ✅ Comoo opción alternativa, se puede usar DVC Experiments (`dvc exp`), probando `dataset_v2.csv` sin modificar `params.yaml`.

---

## ⚙️ Configuración general

El archivo `params.yaml` controla los parámetros del pipeline:

- Ruta y nombre del dataset (`dataset.path`)
- Columna objetivo (`dataset.target`)
- Opciones de preprocesamiento (imputación, escalado)
- Lista de modelos y sus hiperparámetros
- Número de folds para validación cruzada
- Métrica de evaluación (por defecto: RMSE negativo)

---

## 📦 Salidas generadas

- `artifacts/preprocessed.parquet` — dataset preprocesado  
- `models/best_model.joblib` — mejor modelo entrenado  
- `artifacts/metrics.json` — métricas de evaluación (RMSE, MAE, R²)  
- `reports/report.md` — informe resumen de resultados  
