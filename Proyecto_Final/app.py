import gradio as gr
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import json
import os

# ── Cargar configuración del modelo ──────────────────────────────────────────
with open("model_config.json", "r") as f:
    config = json.load(f)

CLASES     = config["clases"]           # ['benign', 'malignant', 'non-melanoma']
IMG_SIZE   = config["img_size"]         # 300
MEAN       = config["imagenet_mean"]
STD        = config["imagenet_std"]
NUM_CLASSES = config["num_classes"]     # 3

# ── Reconstruir la arquitectura EfficientNet-B3 ───────────────────────────────
# Debe ser idéntica a la definida en Colab
def build_model():
    model = models.efficientnet_b3(weights=None)  # Sin descargar pesos de ImageNet
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, NUM_CLASSES)
    )
    return model

# ── Cargar pesos entrenados ───────────────────────────────────────────────────
DEVICE = torch.device("cpu")  # Hugging Face CPU basic no tiene GPU
model = build_model()
model.load_state_dict(
    torch.load("melanoma_efficientnet_b3.pth", map_location=DEVICE)
)
model.eval()

# ── Transformaciones de preprocesamiento (igual que en Colab) ─────────────────
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

# ── Descripciones clínicas para cada clase ────────────────────────────────────
DESCRIPCIONES = {
    "benign": (
        "🟢 Lesión benigna",
        "La lesión presenta características compatibles con un nevo melanocítico benigno. "
        "No se observan patrones de alta sospecha. Se recomienda seguimiento dermatológico "
        "de rutina anual."
    ),
    "malignant": (
        "🔴 Lesión maligna (melanoma)",
        "La lesión presenta características compatibles con melanoma maligno. "
        "Se recomienda consulta dermatológica URGENTE para evaluación clínica "
        "y posible biopsia. Este resultado no reemplaza el diagnóstico médico profesional."
    ),
    "non-melanoma": (
        "🟡 Cáncer de piel no melanoma",
        "La lesión presenta características compatibles con carcinoma basocelular o "
        "espinocelular. Requiere evaluación dermatológica para confirmación y "
        "planificación de tratamiento. Generalmente de buen pronóstico si se trata a tiempo."
    )
}

# ── Función principal de predicción ──────────────────────────────────────────
def clasificar(imagen):
    if imagen is None:
        return None, "⚠️ Por favor sube una imagen para clasificar."

    # Preprocesar la imagen
    img = imagen.convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)  # Añadir dimensión de batch

    # Inferencia
    with torch.no_grad():
        outputs = model(tensor)
        probs   = torch.softmax(outputs, dim=1)[0]

    # Construir diccionario de probabilidades para Gradio
    resultados = {clase: float(probs[i]) for i, clase in enumerate(CLASES)}

    # Clase predicha
    idx_pred  = probs.argmax().item()
    clase_pred = CLASES[idx_pred]
    confianza  = float(probs[idx_pred]) * 100

    # Descripción clínica
    titulo, descripcion = DESCRIPCIONES[clase_pred]

    mensaje = (
        f"### {titulo}\n\n"
        f"**Confianza del modelo:** {confianza:.1f}%\n\n"
        f"{descripcion}\n\n"
        f"---\n"
        f"⚠️ *Este sistema es una herramienta de apoyo académico. "
        f"No reemplaza el diagnóstico de un dermatólogo certificado.*"
    )

    return resultados, mensaje


# ── Interfaz Gradio ───────────────────────────────────────────────────────────
with gr.Blocks(title="Clasificador de Lesiones Cutáneas") as demo:

    gr.Markdown("""
    # 🔬 Clasificador de Lesiones Cutáneas
    ### EfficientNet-B3 · Transfer Learning · Dataset ISIC (9,358 imágenes)

    Sube una imagen dermoscópica y el modelo clasificará la lesión en una de tres categorías.
    Desarrollado como proyecto académico para el curso de **Procesamiento Digital de Imágenes**.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            imagen_input = gr.Image(
                type="pil",
                label="Imagen de la lesión cutánea",
                height=300
            )
            btn = gr.Button("Clasificar", variant="primary", size="lg")

            gr.Examples(
                examples=[],  # El profesor subirá sus propias imágenes
                inputs=imagen_input
            )

        with gr.Column(scale=1):
            prob_output = gr.Label(
                num_top_classes=3,
                label="Probabilidades por clase"
            )
            texto_output = gr.Markdown(
                label="Interpretación clínica"
            )

    btn.click(
        fn=clasificar,
        inputs=imagen_input,
        outputs=[prob_output, texto_output]
    )

    # También clasificar al subir la imagen directamente
    imagen_input.change(
        fn=clasificar,
        inputs=imagen_input,
        outputs=[prob_output, texto_output]
    )

    gr.Markdown("""
    ---
    **Clases:**
    - 🟢 **Benign** — Nevo melanocítico benigno
    - 🔴 **Malignant** — Melanoma maligno
    - 🟡 **Non-melanoma** — Carcinoma basocelular / espinocelular

    **Métricas del modelo:** Accuracy 91.2% · F1-macro 91.6% · AUC-ROC 98.3%

    *Dataset: [melanoma-b2rp6](https://universe.roboflow.com/shawn-f/melanoma-b2rp6) (Roboflow Universe · CC BY 4.0)*
    """)

demo.launch()
