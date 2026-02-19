from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model


# Carrega e prepara uma imagem para o modelo
def load_image_for_model(
    img_path: str,
    target_size: Tuple[int, int] = (224, 224),
    preprocess_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Tuple[np.ndarray, tf.Tensor]:
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


# Acha automaticamente a última camada com saída 4D (já olhando camadas aninhadas)
def find_last_conv_layer_name(model: tf.keras.Model) -> str:
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


# Procura camada pelo nome usando o grafo “flattened”
def _get_layer_by_name(model: tf.keras.Model, layer_name: str):
    for layer in model._flatten_layers(include_self=False):
        if layer.name == layer_name:
            return layer
    raise ValueError(f"Camada '{layer_name}' não encontrada no modelo.")


# Calcula o heatmap Grad-CAM para uma imagem (batch 1)
def compute_gradcam_heatmap(
    model: tf.keras.Model,
    input_tensor: tf.Tensor,
    last_conv_layer_name: Optional[str] = None,
    class_index: Optional[int] = None,
) -> np.ndarray:
    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer_name(model)

    target_layer = _get_layer_by_name(model, last_conv_layer_name)

    grad_model = Model(
        inputs=model.inputs,
        outputs=[target_layer.output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(input_tensor, training=False)

        # binário ([..., 1]) → índice 0; caso multi-classe, usa argmax
        if class_index is None:
            if preds.shape[-1] == 1:
                class_index = 0
            else:
                class_index = int(tf.argmax(preds[0]))

        class_channel = preds[:, class_index]
        grads = tape.gradient(class_channel, conv_outputs)

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


# Sobrepõe o heatmap na imagem original
def superimpose_heatmap(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
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


# Função principal: carrega, calcula Grad-CAM e devolve tudo
def run_gradcam_on_image(
    model: tf.keras.Model,
    img_path: str,
    target_size: Tuple[int, int] = (224, 224),
    preprocess_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    last_conv_layer_name: Optional[str] = None,
    class_index: Optional[int] = None,
    alpha: float = 0.4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
