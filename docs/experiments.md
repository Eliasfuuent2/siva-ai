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

---

## EXP-03 — Fusión Roboflow + andrewmvd (RESULTADO NEGATIVO)

**Fecha:** 2026-07-21
**Hipótesis:** añadir 5.785 ejemplos de NO-Hardhat del dataset andrewmvd
(Safety Helmet Detection) elevaría el mAP50 de NO-Hardhat, la clase débil
de EXP-02 (0.574 en test).

**Setup:**
- Modelo: yolov8s (misma config que EXP-02: 50 ep, batch 16, imgsz 640,
  mosaic 0.5, seed 42, patience 15).
- Datos: train fusionado = Roboflow train (2.605) + andrewmvd (5.000) = 7.605 imgs.
- Valid (114) y test (82): SOLO Roboflow, intactos → comparación limpia vs EXP-02.
- Mapeo VOC→YOLO: helmet→Hardhat (0), head→NO-Hardhat (2), person→descartado
  (sub-anotado: solo 751 boxes en 5.000 imgs con gente sin anotar).
- Conversión VOC→YOLO validada visualmente (Celdas 3 y 5).

**Resultados sobre TEST (82 imgs, mismo test que EXP-02):**

| Métrica / clase | EXP-02 | EXP-03 | Δ |
|---|---|---|---|
| mAP50 (all) | 0.792 | 0.733 | −0.059 |
| mAP50-95 | 0.488 | 0.437 | −0.051 |
| NO-Hardhat | 0.574 | 0.508 | −0.066 |
| Hardhat | 0.945 | 0.873 | −0.072 |
| Person | 0.894 | 0.791 | −0.103 |
| NO-Safety Vest | 0.825 | 0.683 | −0.142 |

**Veredicto: la fusión degradó el desempeño en todas las clases, incluida
NO-Hardhat, la que buscaba mejorar. Hipótesis refutada.**

**Análisis de causa:**
1. Incompatibilidad de dominio (causa dominante): andrewmvd usa estilo de
   anotación distinto (imgs 416px con reflection padding, criterio de box
   sobre cabeza diferente al de Roboflow). El modelo recibió señales
   contradictorias sobre "cabeza sin casco".
2. Desbalance masivo: +18.966 Hardhat (train pasó a 22.111) desestabilizó
   el equilibrio afinado de EXP-02.
3. Person −0.103: ruido de personas visibles sin anotar en andrewmvd,
   riesgo declarado antes de fusionar y confirmado.

**Decisión:** se descarta la fusión. EXP-02 (yolov8s, solo Roboflow) se
mantiene como MODELO BASE del proyecto. Aprendizaje: para PPE, la calidad y
homogeneidad de anotación pesa más que el volumen de datos.

**Notebook:** 03-fusion-datasets (Kaggle, versión "EXP-03 fusion andrewmvd - resultado negativo").