# Registro de experimentos — SIVA AI

## EDA dataset Roboflow (2026-07-17)
- Dataset: Construction Site Safety (snehilsanyal, Kaggle), 2.801 imágenes
  (train 2605 / valid 114 / test 82), 10 clases, formato YOLOv8.
- Clases críticas bien representadas: NO-Hardhat 2.317, NO-Safety Vest 3.962 (train).
- Alerta: valid/test pequeños (114/82 imgs) → métricas con ruido, declararlo al reportar.
- El train viene con augmentation pre-aplicada (mosaic + cutout de Roboflow).
  Considerar reducir augmentation en el entrenamiento de F1.
- Decisión pendiente F1: entrenar 10 clases y filtrar Mask/NO-Mask/Safety Cone
  en inferencia (inclinación actual), o reprocesar a 7 clases.
- Notebook: notebooks/01-eda-construction-safety.ipynb