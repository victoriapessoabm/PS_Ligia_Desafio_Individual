# Interpretability/gradcam.py

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Callable, Optional, Tuple


# -----------------------------------------------------------------------------
# 1. Carregar imagem para Grad-CAM
# -----------------------------------------------------------------------------
def load_image_for_gradcam(
    img_path: str | Path,
    img_size: int = 224,
    preprocess_fn: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
) -> Tuple[np.ndarray, tf.Tensor]:
    """
    Carrega uma imagem e prepara:
      - original_img: np.ndarray (H, W, 3) em [0,1], só para visualização;
      - input_tensor: tf.Tensor (1, img_size, img_size, 3), pronto para o modelo.
    """
    img_path = Path(img_path)
    if preprocess_fn is None:
        preprocess_fn = tf.keras.applications.efficientnet.preprocess_input

    raw = tf.io.read_file(str(img_path))
    img = tf.image.decode_jpeg(raw, channels=3)
    img = tf.image.resize_with_pad(img, img_size, img_size)

    original_img = tf.cast(img, tf.float32) / 255.0
    original_img_np = original_img.numpy()

    model_input = tf.cast(img, tf.float32)
    model_input = preprocess_fn(model_input)
    model_input = tf.expand_dims(model_input, axis=0)

    return original_img_np, model_input


# -----------------------------------------------------------------------------
# 2. Grad-CAM usando a saída do layer "efficientnetb0"
# -----------------------------------------------------------------------------
def make_gradcam_heatmap(
    input_tensor: tf.Tensor,
    model: tf.keras.Model,
    class_index: Optional[int] = None,
) -> np.ndarray:
    """
    Grad-CAM para o modelo:

        input_layer_1
          -> efficientnetb0          (saída: feature map 7x7x1280, chamada top_activation)
          -> global_average_pooling2d
          -> dropout
          -> dense

    Usa-se a SAÍDA do layer "efficientnetb0" como feature map para Grad-CAM.
    Isso evita misturar grafos internos do backbone com o grafo externo.
    """

    # Camada de feature map (já está no grafo principal do modelo)
    feature_layer = model.get_layer("efficientnetb0")

    # grad_model: entrada -> (feature maps, saída final)
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[feature_layer.output, model.output],
    )

    with tf.GradientTape() as tape:
        # Faz forward uma vez
        conv_outputs, predictions = grad_model(input_tensor, training=False)

        # Escolhe a saída alvo
        if class_index is None:
            if predictions.shape[-1] == 1:
                class_channel = predictions[:, 0]
            else:
                class_channel = tf.reduce_max(predictions, axis=-1)
        else:
            class_channel = predictions[:, class_index]

        # Gradientes da saída alvo em relação às ativações da feature_layer
        grads = tape.gradient(class_channel, conv_outputs)

    # Média espacial dos gradientes
    pooled_grads = tf.reduce_mean(grads, axis=(1, 2))  # (1, C)
    conv_outputs = conv_outputs[0]                     # (H_feat, W_feat, C)
    pooled_grads = pooled_grads[0]                     # (C,)

    # Pondera cada canal pela importância
    conv_outputs = conv_outputs * pooled_grads

    heatmap = tf.reduce_sum(conv_outputs, axis=-1)     # (H_feat, W_feat)
    heatmap = tf.nn.relu(heatmap)

    max_val = tf.reduce_max(heatmap)
    heatmap = tf.where(max_val > 0, heatmap / max_val, tf.zeros_like(heatmap))

    return heatmap.numpy()


# -----------------------------------------------------------------------------
# 3. Overlay do heatmap na imagem original
# -----------------------------------------------------------------------------
def overlay_gradcam(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    cmap: str = "jet",
) -> np.ndarray:
    """
    Faz o overlay do heatmap Grad-CAM na imagem original.
    """
    import matplotlib.cm as cm

    h, w, _ = original_image.shape
    heatmap_resized = tf.image.resize(
        heatmap[..., np.newaxis], (h, w)
    ).numpy().squeeze()

    colormap = cm.get_cmap(cmap)
    heatmap_rgb = colormap(heatmap_resized)[:, :, :3]

    original = np.clip(original_image, 0.0, 1.0)
    superimposed = (1 - alpha) * original + alpha * heatmap_rgb
    superimposed = np.clip(superimposed, 0.0, 1.0)

    return superimposed


# -----------------------------------------------------------------------------
# 4. Plot do resultado
# -----------------------------------------------------------------------------
def plot_gradcam_result(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    superimposed_image: np.ndarray,
    title: str = "",
    prob: Optional[float] = None,
    true_label: Optional[str] = None,
):
    plt.figure(figsize=(10, 3))

    plt.subplot(1, 3, 1)
    plt.imshow(original_image)
    plt.axis("off")
    plt.title("Imagem original")

    plt.subplot(1, 3, 2)
    plt.imshow(heatmap, cmap="jet")
    plt.axis("off")
    plt.title("Heatmap Grad-CAM")

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
