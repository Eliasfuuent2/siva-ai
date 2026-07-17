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

## EXP-01: yolov8n línea base (2026-07-17)
- Modelo: yolov8n.pt (COCO) fine-tuned | 50 épocas (completó las 50), batch 16,
  imgsz 640, mosaic 0.5, seed 42 | GPU T4 | duración 0.4 h
- Dataset: Roboflow css-data sin modificar (train 2605 / valid 114 / test 82)
- TEST: mAP50 0.736 | mAP50-95 0.422 | P 0.850 | R 0.666
- Por clase (mAP50 test): Hardhat 0.883, NO-Hardhat 0.528 ⚠, Safety Vest 0.874,
  NO-Safety Vest 0.774, Person 0.830, machinery 0.854, vehicle 0.721
- Conclusión: base sólida; debilidad crítica en NO-Hardhat (recall 0.56).
  Acciones: EXP-02 yolov8s; evaluar fusión dataset andrewmvd para reforzar cascos.
- Notebook: 02-finetune-yolov8n (Kaggle)

## EXP-02: yolov8s misma config (2026-07-17)
- Modelo: yolov8s.pt fine-tuned | idénticos hiperparámetros a EXP-01 | 0.64 h GPU
- TEST: mAP50 0.792 | mAP50-95 0.488 | P 0.900 | R 0.740
- Por clase clave (mAP50): Hardhat 0.945, NO-Hardhat 0.574 ⚠, Safety Vest 0.894,
  NO-Safety Vest 0.825, Person 0.894, machinery 0.907, vehicle 0.818
- Conclusión: v8s > v8n en todas las métricas (+5.6 mAP50); queda como modelo base.
  NO-Hardhat mejora poco (+4.6) → el límite son los datos, no el modelo.
  Próximo: EXP-03 fusión con dataset andrewmvd (helmet/head) para reforzar NO-Hardhat.
- Inferencia: 11.5 ms/img en T4 (apto tiempo real)
- Notebook: 02-finetune-yolov8n (Kaggle)