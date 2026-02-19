# Interpretability/gradcam.py

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Callable, Optional, Tuple


# -----------------------------------------------------------------------------
# 0. Helpers de detecção de camadas
# -----------------------------------------------------------------------------
def _is_conv_like(layer: tf.keras.layers.Layer) -> bool:
    """
    Considera como 'conv-like' apenas camadas convolucionais de verdade.
    """
    conv_types = (
        tf.keras.layers.Conv2D,
        tf.keras.layers.SeparableConv2D,
        tf.keras.layers.DepthwiseConv2D,
    )
    return isinstance(layer, conv_types)


def find_last_conv_layer(model: tf.keras.Model) -> tf.keras.layers.Layer:
    """
    Procura, em ordem reversa, a última camada convolucional do modelo,
    incluindo convoluções que estejam dentro de submodelos (como EfficientNet).
    """
    for layer in reversed(model.layers):
        # Se for um submodelo (ex.: efficientnetb0), varre as camadas internas
        if isinstance(layer, tf.keras.Model) and hasattr(layer, "layers"):
            for sub_layer in reversed(layer.layers):
                if _is_conv_like(sub_layer):
                    return sub_layer

        if _is_conv_like(layer):
            return layer

    raise ValueError(
        "Não foi possível encontrar uma camada convolucional adequada "
        "para Grad-CAM. Considere passar manualmente a camada."
    )


# -----------------------------------------------------------------------------
# 2. Carregamento e pré-processamento de imagem
# -----------------------------------------------------------------------------
def load_image_for_gradcam(
    img_path: str | Path,
    img_size: int = 224,
    preprocess_fn: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
) -> Tuple[np.ndarray, tf.Tensor]:
    """
    Carrega uma imagem de disco e prepara duas versões:

    - original_img: imagem em [0,1], formato (H, W, 3), apenas para visualização;
    - input_tensor: tensor (1, img_size, img_size, 3), pronto para o modelo.
    """
    img_path = Path(img_path)
    if preprocess_fn is None:
        preprocess_fn = tf.keras.applications.efficientnet.preprocess_input

    raw = tf.io.read_file(str(img_path))
    img = tf.image.decode_jpeg(raw, channels=3)
    img = tf.image.resize_with_pad(img, img_size, img_size)

    # Versão para visualização (0-1)
    original_img = tf.cast(img, tf.float32) / 255.0
    original_img_np = original_img.numpy()

    # Versão para o modelo
    model_input = tf.cast(img, tf.float32)
    model_input = preprocess_fn(model_input)
    model_input = tf.expand_dims(model_input, axis=0)  # (1, H, W, 3)

    return original_img_np, model_input


# -----------------------------------------------------------------------------
# 3. Cálculo do heatmap Grad-CAM
# -----------------------------------------------------------------------------
def make_gradcam_heatmap(
    input_tensor: tf.Tensor,
    model: tf.keras.Model,
    last_conv_layer_name: Optional[str] = None,
    class_index: Optional[int] = None,
) -> np.ndarray:
    """
    Gera o heatmap Grad-CAM para um único exemplo (batch size = 1).

    Implementação específica para o modelo:

        input_layer_1
          -> efficientnetb0
          -> global_average_pooling2d
          -> dropout
          -> dense

    Dentro do efficientnetb0, a última conv é a `top_conv`, seguida de:
        top_bn -> top_activation

    A cadeia exata usada aqui é:
        input -> ... -> top_conv -> top_bn -> top_activation
        -> global_average_pooling2d -> dropout -> dense
    """

    # 1) Descobre a última camada conv (no seu caso, top_conv)
    last_conv_layer = find_last_conv_layer(model)  # deve ser 'top_conv'

    # 2) Modelo que vai da entrada até a última camada conv
    conv_model = tf.keras.Model(model.inputs, last_conv_layer.output)

    # 3) Monta o "classificador" a partir de top_conv até a saída final
    #    usando as camadas já existentes (pesos compartilhados).
    backbone = model.get_layer("efficientnetb0")
    top_bn = backbone.get_layer("top_bn")
    top_activation = backbone.get_layer("top_activation")

    gap = model.get_layer("global_average_pooling2d")
    drop = model.get_layer("dropout")
    dense = model.get_layer("dense")

    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    x = top_bn(x)
    x = top_activation(x)
    x = gap(x)
    x = drop(x)
    x = dense(x)
    classifier_model = tf.keras.Model(classifier_input, x)

    # 4) Grad-CAM propriamente dito
    with tf.GradientTape() as tape:
        # Feature maps da última conv
        conv_outputs = conv_model(input_tensor)
        tape.watch(conv_outputs)

        # Predição final passando pelos "restantes" do modelo
        predictions = classifier_model(conv_outputs)

        # Saída alvo
        if class_index is None:
            if predictions.shape[-1] == 1:
                class_channel = predictions[:, 0]
            else:
                class_channel = tf.reduce_max(predictions, axis=-1)
        else:
            class_channel = predictions[:, class_index]

        # Gradientes da saída alvo em relação às ativações da última conv
        grads = tape.gradient(class_channel, conv_outputs)

    # 5) Pooling dos gradientes e combinação linear dos feature maps
    pooled_grads = tf.reduce_mean(grads, axis=(1, 2))  # (1, C)
    conv_outputs = conv_outputs[0]  # (H_feat, W_feat, C)
    pooled_grads = pooled_grads[0]  # (C,)

    conv_outputs = conv_outputs * pooled_grads

    heatmap = tf.reduce_sum(conv_outputs, axis=-1)  # (H_feat, W_feat)
    heatmap = tf.nn.relu(heatmap)

    # 6) Normaliza para [0, 1]
    max_val = tf.reduce_max(heatmap)
    heatmap = tf.where(max_val > 0, heatmap / max_val, tf.zeros_like(heatmap))

    return heatmap.numpy()


# -----------------------------------------------------------------------------
# 4. Overlay do heatmap na imagem original
# -----------------------------------------------------------------------------
def overlay_gradcam(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    cmap: str = "jet",
) -> np.ndarray:
    """
    Faz o overlay do heatmap Grad-CAM sobre a imagem original.
    """
    import matplotlib.cm as cm

    # Redimensiona heatmap para o tamanho da imagem
    h, w, _ = original_image.shape
    heatmap_resized = tf.image.resize(
        heatmap[..., np.newaxis], (h, w)
    ).numpy().squeeze()

    # Converte heatmap para RGB via colormap
    colormap = cm.get_cmap(cmap)
    heatmap_rgb = colormap(heatmap_resized)[:, :, :3]  # descarta canal alpha

    # Normaliza original
    original = np.clip(original_image, 0.0, 1.0)

    # Combina
    superimposed_img = (1 - alpha) * original + alpha * heatmap_rgb
    superimposed_img = np.clip(superimposed_img, 0.0, 1.0)

    return superimposed_img


# -----------------------------------------------------------------------------
# 5. Helper para plotar o resultado completo
# -----------------------------------------------------------------------------
def plot_gradcam_result(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    superimposed_image: np.ndarray,
    title: str = "",
    prob: Optional[float] = None,
    true_label: Optional[str] = None,
):
    """
    Faz um plot com três painéis:
    - imagem original;
    - heatmap sozinho;
    - overlay Grad-CAM.
    """
    plt.figure(figsize=(10, 3))

    # Original
    plt.subplot(1, 3, 1)
    plt.imshow(original_image)
    plt.axis("off")
    plt.title("Imagem original")

    # Heatmap
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap, cmap="jet")
    plt.axis("off")
    plt.title("Heatmap Grad-CAM")

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
    if title:
        if subtitle:
            final_title = f"{title}\n{subtitle}"
        else:
            final_title = title
    else:
        final_title = subtitle

    plt.title(final_title, fontsize=9)
    plt.tight_layout()
    plt.show()
