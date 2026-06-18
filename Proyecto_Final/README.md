# Clasificador de Lesiones Cutáneas

Proyecto académico para la asignatura **Procesamiento Digital de Imágenes**. Sistema de clasificación de imágenes dermoscópicas en tres categorías clínicas, basado en transfer learning con EfficientNet-B3.

**Demo en vivo:** [Hugging Face Space](https://huggingface.co/spaces/hpiarpuezan/melanoma-classifier)

---

## 1. Contexto del problema

El diagnóstico temprano de melanoma es crítico. Si se detecta a tiempo tiene tasas de supervivencia altas pero un diagnóstico tardío reduce drásticamente ese pronóstico. En la práctica clínica, los dermatólogos deben priorizar qué lesiones requieren biopsia urgente entre un alto volumen de consultas.

Este proyecto plantea un **sistema de apoyo visual**: un clasificador que, a partir de una imagen dermoscópica, distingue entre lesión benigna, melanoma maligno y otros cánceres de piel no-melanoma (carcinoma basocelular/espinocelular). La tarea de visión por computador empleada es **clasificación de imágenes** (no detección ni segmentación), ya que el objetivo es asignar una etiqueta global a la imagen completa de la lesión, previamente recortada/centrada por el propio proceso de adquisición dermoscópica.

> ⚠️ Este sistema es una herramienta de apoyo académico. No reemplaza el diagnóstico de un dermatólogo certificado en la materia.

---

## 2. Base de datos

| Campo | Detalle |
|---|---|
| **Nombre** | `melanoma` |
| **Fuente** | [Roboflow Universe](https://app.roboflow.com/robox-b7rra/melanoma-b2rp6-oa0tf/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true) |
| **Origen de las imágenes** | Dataset ISIC (International Skin Imaging Collaboration) |
| **Licencia** | CC BY 4.0 |
| **Tamaño total** | 9,358 imágenes |
| **Clases** | `benign`, `malignant`, `non-melanoma` |
| **Resolución de exportación** | 640×640 px |

### Distribución original vs. redistribuida

El dataset descargado desde Roboflow venía particionado 84% train / 12% valid / 4% test, con la totalidad de las 412 imágenes de `non-melanoma` asignadas exclusivamente al split de entrenamiento (0 imágenes de esta clase en valid/test). Dado que el conjunto de test resultante era insuficiente (~375 imágenes) y no representaba a todas las clases, se realizó una redistribución estratificada 70/15/15 sobre el total de imágenes:

| Split | Total | benign | malignant | non-melanoma |
|---|---|---|---|---|
| Train | 6,550 | 3,029 | 3,109 | 412 |
| Validación | 1,404 | 649 | 666 | 89 |
| Test | 1,404 | 650 | 666 | 88 |

**Desbalance de clases:** `non-melanoma` representa solo el 6.3% del dataset (412 vs. ~3,100 de las otras dos clases), un desbalance alto.

### ⚠️ Importante para reproducir pruebas con imágenes nuevas

Las imágenes de la **galería web** de Roboflow Universe (vista previa al navegar el dataset) son thumbnails comprimidos de aproximadamente 200×200 px, de menor calidad que las imágenes de **exportación** (640×640 px) usadas para entrenar el modelo. Esta diferencia de resolución provoca errores de clasificación al usarse como entrada del modelo desplegado.

**Para probar el modelo correctamente:** se descarga el dataset completo usando el botón **"Download Dataset"** de Roboflow (o el script de la sección 3), y usa imágenes de las carpetas `train/valid/test` resultantes — no imágenes guardadas por clic derecho desde la página de exploración del dataset.

---

## 3. Cómo descargar el dataset

```python
pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="TU_API_KEY")  # gratis en roboflow.com
project = rf.workspace("robox-b7rra").project("melanoma-b2rp6-oa0tf")
version = project.version(1)
dataset = version.download("folder")
```

---

## 4. Arquitectura y entrenamiento

| Aspecto | Decisión | Justificación |
|---|---|---|
| **Arquitectura** | EfficientNet-B3 (preentrenado en ImageNet) | Mejor balance accuracy/parámetros (12M params, 81.6% top-1) para GPU T4 con recursos limitados, frente a alternativas como ResNet-50 o VGG-16 |
| **Estrategia** | Transfer learning en 2 fases | Fase 1 (10 épocas): solo se entrena la cabeza de clasificación, backbone congelado. Fase 2 (15 épocas): fine-tuning de las últimas capas del backbone con LR reducido |
| **Función de pérdida** | `CrossEntropyLoss` | Estándar para clasificación multiclase con salida softmax |
| **Hiperparámetros** | Batch size 32, LR Fase 1 = 1e-3, LR Fase 2 = 1e-4, weight decay = 1e-4 | Ver detalle completo en notebook |

### Métricas de rendimiento utilizadas

- **Accuracy, Precision, Recall, F1-score (macro y por clase):** dado el desbalance de clases, el F1-macro y el recall por clase son más informativos que el accuracy global — en contexto clínico, el recall de la clase `malignant` es la métrica más crítica, porque un falso negativo (melanoma clasificado como benigno) tiene el mayor costo posible.

- **AUC-ROC (One-vs-Rest):** mide la capacidad de separación del modelo independientemente del umbral de decisión, útil para verificar que el modelo discrimina bien entre clases incluso en la minoritaria.

---
## 5. Despliegue en Hugging Face Spaces

El modelo está desplegado como una aplicación Gradio en Hugging Face Spaces. Los archivos que lo componen son los siguientes:

* **`app.py`**: contiene la lógica de la aplicación. Reconstruye la arquitectura de EfficientNet-B3 (idéntica a la del entrenamiento), carga los pesos entrenados, aplica el preprocesamiento a la imagen de entrada y genera la predicción junto con una interpretación clínica textual mostrada en la interfaz de Gradio.

* **`melanoma_efficientnet_b3.pth`**: pesos del modelo ya entrenado (exportados desde Colab en el archivo melanoma_clasificacion.ipynb), cargados por `app.py` al iniciar el Space. (Este archivo no esta cargado aqui en github debido a que excede el limite de peso de 25 MB).

* **`model_config.json`**: archivo de configuración con los metadatos necesarios para reconstruir el modelo correctamente que son nombre de las clases, mapeo `class_to_idx`, tamaño de imagen esperado (`img_size`) y los valores de media/desviación estándar de ImageNet usados en la normalización.

* **`requirements.txt`**: lista las dependencias de Python necesarias para que el Space funcione (versiones de `torch`, `torchvision` y `Pillow` compatibles con el entorno de Hugging Face).

* **`melanoma_clasificacion.ipynb`**: Este es el archivo donde esta el codigo fuente del entrenamiento en google colab. 

La aplicación está configurada de la siguiente manera:

- **SDK:** Gradio
- **Hardware:** CPU basic
- **Entrada:** imagen (PIL), redimensionada a 300×300 px y normalizada con estadísticas de ImageNet
- **Salida:** probabilidades por clase + interpretación clínica textual

Además, tambien implemetamos un dashboard más interactivo y educativo donde se explica a mayor profundidad lo que significa cada clase clinicamente, con ejemplos y el clasificador integrado en el siguiente link:
[Dashboard](https://huggingface.co/spaces/hpiarpuezan/melanoma-app-educativa)

---

## 6. Limitaciones conocidas

- El tamaño reducido de la clase `non-melanoma` (412 imágenes) limita la capacidad de generalización del modelo a imágenes externas al dataset ISIC.

- Las métricas de test pueden estar parcialmente infladas por similitud de origen entre las imágenes de entrenamiento y test de `non-melanoma`, dado que esta clase no existía en los splits originales de validación/test del dataset descargado.
  
- El modelo es sensible a la resolución/calidad de la imagen de entrada; imágenes de baja resolución (p. ej. thumbnails) pueden producir predicciones incorrectas.

---

## 7. Estructura del repositorio

```
Proyecto_Final/
├── Presentación proyecto Final.pptx
├── README.md
├── app.py
├── melanoma_clasificacion.ipynb
├── model_config.json
├── requirements.txt/

```
