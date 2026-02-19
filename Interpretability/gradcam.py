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
    (Não usa mais o truque de rank 4 para não cair em top_activation.)
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
        "para Grad-CAM. Considere passar manualmente o nome da camada."
    )


def _get_layer_recursive(model: tf.keras.Model, name: str) -> tf.keras.layers.Layer:
    """
    Procura uma camada por nome dentro do modelo, incluindo submodelos aninhados.
    Permite, por exemplo, achar 'top_conv' dentro de 'efficientnetb0'.
    """
    for layer in model.layers:
        if layer.name == name:
            return layer
        if isinstance(layer, tf.keras.Model):
            try:
                return _get_layer_recursive(layer, name)
            except ValueError:
                # Se não estiver nesse submodelo, continua procurando em outros
                pass
    raise ValueError(f"No such layer: {name}")


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

    Por padrão, usa tf.keras.applications.efficientnet.preprocess_input.
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

    Parâmetros
    ----------
    input_tensor:
        Tensor de entrada no formato (1, H, W, 3), já pré-processado.
    model:
        Modelo Keras já carregado (por exemplo, best_model.keras).
    last_conv_layer_name:
        Nome da camada convolucional final a ser usada para Grad-CAM.
        Se None, é detectada automaticamente com find_last_conv_layer.
    class_index:
        Índice da classe alvo para o Grad-CAM.
        - Binário (Dense(1, sigmoid)): use 0.
        - Multiclasse (Dense(C, softmax)): pode ser None para usar argmax.

    Retorno
    -------
    heatmap: np.ndarray, shape (H_feat, W_feat), com valores em [0, 1].
    """
    # Decide qual camada convolucional usar
    if last_conv_layer_name is None:
        last_conv_layer = find_last_conv_layer(model)
    else:
        # Busca recursiva: funciona mesmo se a camada estiver dentro de submodelo
        last_conv_layer = _get_layer_recursive(model, last_conv_layer_name)

    # Modelo que mapeia input -> (feature maps da última conv, saída final)
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [last_conv_layer.output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(input_tensor)
        # predictions tem shape (1, 1) ou (1, C)
        if class_index is None:
            if predictions.shape[-1] == 1:
                # Caso binário: usar a única saída
                class_channel = predictions[:, 0]
            else:
                # Multiclasse: usa classe com maior probabilidade
                class_channel = tf.reduce_max(predictions, axis=-1)
        else:
            class_channel = predictions[:, class_index]

        # Gradientes da classe alvo em relação às ativações da última conv
        grads = tape.gradient(class_channel, conv_outputs)

    # Média espacial dos gradientes (importância de cada filtro)
    pooled_grads = tf.reduce_mean(grads, axis=(1, 2))  # (1, C)
    conv_outputs = conv_outputs[0]  # (H_feat, W_feat, C)
    pooled_grads = pooled_grads[0]  # (C,)

    # Combinação linear dos mapas de ativação ponderados pelos gradientes
    conv_outputs = conv_outputs * pooled_grads

    heatmap = tf.reduce_sum(conv_outputs, axis=-1)  # (H_feat, W_feat)
    heatmap = tf.nn.relu(heatmap)

    # Normaliza para [0, 1]
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

    Parâmetros
    ----------
    original_image:
        np.ndarray (H, W, 3) em [0, 1], como retornado por load_image_for_gradcam.
    heatmap:
        np.ndarray (H_feat, W_feat) em [0, 1].
    alpha:
        Peso do heatmap. Valores mais altos destacam mais o mapa de calor.
    cmap:
        Colormap Matplotlib (ex.: "jet", "viridis").

    Retorno
    -------
    superimposed_img: np.ndarray (H, W, 3) em [0, 1].
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
