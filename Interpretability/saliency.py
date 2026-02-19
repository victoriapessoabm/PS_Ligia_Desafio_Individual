# Interpretability/saliency.py

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Callable, Optional, Tuple


# -----------------------------------------------------------------------------
# 1. Carregar imagem para Saliency Map
# -----------------------------------------------------------------------------
def load_image_for_saliency(
    img_path: str | Path,
    img_size: int = 224,
    preprocess_fn: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
) -> Tuple[np.ndarray, tf.Tensor]:
    """
    Carrega uma imagem e prepara:
      - original_img: np.ndarray (H, W, 3) em [0,1], para visualização;
      - input_tensor: tf.Tensor (1, img_size, img_size, 3), pronto para o modelo.
    """
    img_path = Path(img_path)
    if preprocess_fn is None:
        preprocess_fn = tf.keras.applications.efficientnet.preprocess_input

    raw = tf.io.read_file(str(img_path))
    img = tf.image.decode_jpeg(raw, channels=3)
    img = tf.image.resize_with_pad(img, img_size, img_size)

    # Para visualização
    original_img = tf.cast(img, tf.float32) / 255.0
    original_img_np = original_img.numpy()

    # Para o modelo
    model_input = tf.cast(img, tf.float32)
    model_input = preprocess_fn(model_input)
    model_input = tf.expand_dims(model_input, axis=0)  # (1, H, W, 3)

    return original_img_np, model_input


# -----------------------------------------------------------------------------
# 2. Saliency Map (gradiente da saída em relação ao input)
# -----------------------------------------------------------------------------
def compute_saliency_map(
    input_tensor: tf.Tensor,
    model: tf.keras.Model,
    class_index: Optional[int] = None,
    use_absolute: bool = True,
) -> np.ndarray:
    """
    Calcula o saliency map de um exemplo (batch size = 1), a partir do
    gradiente da saída alvo em relação aos pixels de entrada.

    Parâmetros
    ----------
    input_tensor:
        Tensor (1, H, W, 3) já pré-processado.
    model:
        Modelo Keras carregado (ex.: best_model.keras).
    class_index:
        Índice da classe alvo.
        - Binário (Dense(1, sigmoid)): use 0 ou deixe None.
        - Multiclasse (Dense(C, softmax)): pode ser None para usar argmax.
    use_absolute:
        Se True, usa |gradiente|. Do contrário, usa gradiente bruto.

    Retorno
    -------
    saliency: np.ndarray (H, W), normalizado em [0, 1].
    """
    x = tf.convert_to_tensor(input_tensor)

    with tf.GradientTape() as tape:
        tape.watch(x)
        preds = model(x, training=False)

        if class_index is None:
            if preds.shape[-1] == 1:
                class_channel = preds[:, 0]
            else:
                class_channel = tf.reduce_max(preds, axis=-1)
        else:
            class_channel = preds[:, class_index]

        grads = tape.gradient(class_channel, x)  # (1, H, W, 3)

    grads = grads[0]  # (H, W, 3)

    if use_absolute:
        grads = tf.abs(grads)

    # Agrega canais de cor em um único mapa (máximo ao longo de channel)
    saliency = tf.reduce_max(grads, axis=-1)  # (H, W)

    # Normaliza para [0, 1]
    min_val = tf.reduce_min(saliency)
    max_val = tf.reduce_max(saliency)
    denom = tf.maximum(max_val - min_val, tf.constant(1e-8))
    saliency = (saliency - min_val) / denom

    return saliency.numpy()


# -----------------------------------------------------------------------------
# 3. Overlay do saliency na imagem original
# -----------------------------------------------------------------------------
def overlay_saliency(
    original_image: np.ndarray,
    saliency: np.ndarray,
    alpha: float = 0.45,
    cmap: str = "jet",
) -> np.ndarray:
    """
    Combina o mapa de saliency com a imagem original.
    """
    import matplotlib.cm as cm

    h, w, _ = original_image.shape
    saliency_resized = tf.image.resize(
        saliency[..., np.newaxis], (h, w)
    ).numpy().squeeze()

    colormap = cm.get_cmap(cmap)
    saliency_rgb = colormap(saliency_resized)[:, :, :3]

    original = np.clip(original_image, 0.0, 1.0)
    superimposed = (1 - alpha) * original + alpha * saliency_rgb
    superimposed = np.clip(superimposed, 0.0, 1.0)

    return superimposed


# -----------------------------------------------------------------------------
# 4. Plot do resultado
# -----------------------------------------------------------------------------
def plot_saliency_result(
    original_image: np.ndarray,
    saliency: np.ndarray,
    superimposed_image: np.ndarray,
    title: str = "",
    prob: Optional[float] = None,
    true_label: Optional[str] = None,
):
    """
    Plota:
      - imagem original,
      - saliency map,
      - overlay imagem + saliency.
    """
    plt.figure(figsize=(10, 3))

    # Original
    plt.subplot(1, 3, 1)
    plt.imshow(original_image)
    plt.axis("off")
    plt.title("Imagem original")

    # Saliency
    plt.subplot(1, 3, 2)
    plt.imshow(saliency, cmap="jet")
    plt.axis("off")
    plt.title("Saliency map")

    # Overlay
    plt.subplot(1, 3, 3)
    plt.imshow(superimposed_image)
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
