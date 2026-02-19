from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model


def load_image_for_model(
    img_path: str,
    target_size: Tuple[int, int] = (224, 224),
    preprocess_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Tuple[np.ndarray, tf.Tensor]:
    """
    Carrega a imagem do disco, redimensiona e aplica o pré-processamento.
    Retorna (imagem_original_uint8, tensor_pronto_para_modelo).
    """
    img_raw = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img_raw, channels=3)

    img_resized = tf.image.resize(img, target_size)
    original_image = tf.cast(tf.clip_by_value(img_resized, 0, 255), tf.uint8).numpy()

    img_float = tf.cast(img_resized, tf.float32)
    if preprocess_fn is not None:
        img_float = preprocess_fn(img_float)
    else:
        img_float = img_float / 255.0

    input_tensor = tf.expand_dims(img_float, axis=0)  # (1, H, W, 3)
    return original_image, input_tensor


def find_last_conv_layer_name(model: tf.keras.Model) -> str:
    """
    Percorre todas as camadas do grafo (incluindo submodelos) e devolve
    o nome da última camada com saída 4D (HxWxC).
    """
    last_name = None
    for layer in model._flatten_layers(include_self=False):
        try:
            shape = layer.output_shape  # type: ignore[attr-defined]
        except AttributeError:
            continue

        if isinstance(shape, tuple) and len(shape) == 4:
            last_name = layer.name

    if last_name is None:
        raise ValueError("Nenhuma camada convolucional 4D foi encontrada no modelo.")

    return last_name


def _get_layer_by_name(model: tf.keras.Model, layer_name: str):
    """
    Procura camada pelo nome no grafo achatado.
    """
    for layer in model._flatten_layers(include_self=False):
        if layer.name == layer_name:
            return layer
    raise ValueError(f"Camada '{layer_name}' não encontrada no modelo.")


def compute_gradcam_heatmap(
    model: tf.keras.Model,
    input_tensor: tf.Tensor,
    last_conv_layer_name: Optional[str] = None,
    class_index: Optional[int] = None,
) -> np.ndarray:
    """
    Calcula o heatmap Grad-CAM usando a última camada conv 4D do modelo
    (ou uma camada específica, se last_conv_layer_name for fornecido).
    Implementação usando DOIS modelos dentro do mesmo GradientTape
    para evitar problemas de grafo.
    """
    # escolhe camada alvo
    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer_name(model)

    target_layer = _get_layer_by_name(model, last_conv_layer_name)

    # modelo que devolve o mapa da conv
    conv_model = Model(
        inputs=model.inputs,
        outputs=target_layer.output,
    )

    # modelo completo para a predição final
    pred_model = model  # só para deixar explícito

    with tf.GradientTape() as tape:
        # ambos são chamados dentro do mesmo tape
        conv_outputs = conv_model(input_tensor, training=False)   # (1, H, W, C)
        preds = pred_model(input_tensor, training=False)          # (1, num_classes ou 1)

        # binário: shape [..., 1] → índice 0; senão, argmax
        if class_index is None:
            if preds.shape[-1] == 1:
                class_index_used = 0
            else:
                class_index_used = int(tf.argmax(preds[0]))
        else:
            class_index_used = class_index

        class_channel = preds[:, class_index_used]
        grads = tape.gradient(class_channel, conv_outputs)

    # média dos gradientes em HxW
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]  # (H, W, C)

    heatmap = tf.zeros(conv_outputs.shape[0:2], dtype=tf.float32)
    for i in range(conv_outputs.shape[-1]):
        heatmap += pooled_grads[i] * conv_outputs[:, :, i]

    heatmap = tf.nn.relu(heatmap)

    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap /= max_val

    return heatmap.numpy()


def superimpose_heatmap(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Sobrepõe o heatmap na imagem original.
    """
    import matplotlib.cm as cm

    if original_image.dtype != np.uint8:
        original_image = np.clip(original_image, 0, 255).astype("uint8")

    h, w, _ = original_image.shape
    heatmap_resized = tf.image.resize(
        tf.expand_dims(heatmap, axis=-1), (h, w)
    ).numpy()[:, :, 0]

    colormap = cm.get_cmap("jet")
    heatmap_rgb = colormap(heatmap_resized)[:, :, :3]
    heatmap_rgb = np.uint8(255 * heatmap_rgb)

    overlay = np.uint8(original_image * (1 - alpha) + heatmap_rgb * alpha)
    return overlay


def run_gradcam_on_image(
    model: tf.keras.Model,
    img_path: str,
    target_size: Tuple[int, int] = (224, 224),
    preprocess_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    last_conv_layer_name: Optional[str] = None,
    class_index: Optional[int] = None,
    alpha: float = 0.4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pipeline completo:
    - carrega a imagem
    - pré-processa
    - calcula o heatmap Grad-CAM
    - gera o overlay sobre a imagem original
    """
    original_image, input_tensor = load_image_for_model(
        img_path=img_path,
        target_size=target_size,
        preprocess_fn=preprocess_fn,
    )

    heatmap = compute_gradcam_heatmap(
        model=model,
        input_tensor=input_tensor,
        last_conv_layer_name=last_conv_layer_name,
        class_index=class_index,
    )

    overlay = superimpose_heatmap(
        original_image=original_image,
        heatmap=heatmap,
        alpha=alpha,
    )

    return original_image, heatmap, overlay
