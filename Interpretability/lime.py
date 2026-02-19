# Interpretability/lime_explain.py

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Callable, Tuple

from lime import lime_image
from skimage.segmentation import mark_boundaries


# -----------------------------------------------------------------------------
# 1. Preparar imagem para LIME
# -----------------------------------------------------------------------------
def load_image_for_lime(
    img_path: str | Path,
    img_size: int = 224,
) -> np.ndarray:
    """
    Carrega uma imagem de disco e retorna um array uint8 (H, W, 3)
    em [0, 255], adequado para o LIME.
    """
    img_path = Path(img_path)

    raw = tf.io.read_file(str(img_path))
    img = tf.image.decode_jpeg(raw, channels=3)
    img = tf.image.resize_with_pad(img, img_size, img_size)

    img = tf.clip_by_value(img, 0, 255)
    img_uint8 = tf.cast(img, tf.uint8).numpy()

    return img_uint8


# -----------------------------------------------------------------------------
# 2. Função de previsão para o LIME
# -----------------------------------------------------------------------------
def make_lime_predict_fn(
    model: tf.keras.Model,
    img_size: int = 224,
    preprocess_fn: Callable[[tf.Tensor], tf.Tensor] = tf.keras.applications.efficientnet.preprocess_input,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Cria uma função classifier_fn para o LIME.

    A função recebe um array de imagens (N, H, W, 3) em [0, 255] ou [0, 1]
    e retorna um array (N, 2) com [p(classe 0), p(classe 1)].

    Assume modelo binário com saída Dense(1, sigmoid).
    """

    def _predict(images: np.ndarray) -> np.ndarray:
        # Converte para tensor e normaliza para float32
        x = tf.convert_to_tensor(images, dtype=tf.float32)

        # Garante tamanho correto
        x = tf.image.resize_with_pad(x, img_size, img_size)

        # Pré-processamento da EfficientNet
        x = preprocess_fn(x)

        # Predição
        preds = model(x, training=False).numpy()  # (N, 1) ou (N, C)

        # Converte para formato (N, 2): [p0, p1]
        if preds.shape[1] == 1:
            p1 = preds[:, 0]
            p0 = 1.0 - p1
            probs = np.stack([p0, p1], axis=1)
        else:
            probs = preds

        return probs

    return _predict


# -----------------------------------------------------------------------------
# 3. Explicação LIME para uma única imagem
# -----------------------------------------------------------------------------
def explain_with_lime(
    img_uint8: np.ndarray,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    num_samples: int = 1000,
    positive_only: bool = True,
    num_features: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aplica LIME em uma única imagem uint8 (H, W, 3).

    Retorna:
      - image_lime: imagem com os superpixels destacados.
      - mask: máscara booleana dos superpixels mais relevantes.
    """
    explainer = lime_image.LimeImageExplainer()

    explanation = explainer.explain_instance(
        image=img_uint8,
        classifier_fn=predict_fn,
        top_labels=2,
        hide_color=0,
        num_samples=num_samples,
    )

    # Pega o rótulo mais importante segundo o LIME
    label_idx = explanation.top_labels[0]

    image_lime, mask = explanation.get_image_and_mask(
        label=label_idx,
        positive_only=positive_only,
        num_features=num_features,
        hide_rest=False,
    )

    return image_lime, mask


# -----------------------------------------------------------------------------
# 4. Plot do resultado LIME
# -----------------------------------------------------------------------------
def plot_lime_result(
    original_image: np.ndarray,
    image_lime: np.ndarray,
    mask: np.ndarray,
    title: str = "",
    prob: float | None = None,
    true_label: str | None = None,
):
    """
    Plota:
      - imagem original,
      - LIME com bordas dos superpixels relevantes.
    """
    plt.figure(figsize=(8, 3))

    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.axis("off")
    plt.title("Imagem original")

    plt.subplot(1, 2, 2)
    # mark_boundaries desenha contornos dos superpixels em destaque
    lime_vis = mark_boundaries(image_lime / 255.0, mask)
    plt.imshow(lime_vis)
    plt.axis("off")

    subtitle_parts = []
    if true_label is not None:
        subtitle_parts.append(f"rótulo real: {true_label}")
    if prob is not None:
        subtitle_parts.append(f"p(modelo=1)={prob:.3f}")
    subtitle = " | ".join(subtitle_parts)

    final_title = title if not subtitle else f"{title}\n{subtitle}" if title else subtitle
    plt.title(final_title, fontsize=9)

    plt.tight_layout()
    plt.show()
