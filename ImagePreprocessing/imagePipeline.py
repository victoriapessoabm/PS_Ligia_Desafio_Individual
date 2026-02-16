import tensorflow as tf
import pandas as pd
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess

IMG_SIZE = 224

def load_dataframe(csv_path):
    return pd.read_csv(csv_path)

def _decode_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=1)
    img = tf.image.resize_with_pad(img, IMG_SIZE, IMG_SIZE)
    return img

# ========================
# BASELINE PREPROCESS
# ========================

def preprocess_baseline(path, label, base_dir):

    path = tf.strings.join([base_dir, "/", path])
    img = _decode_image(path)

    img = tf.cast(img, tf.float32) / 255.0

    return img, label


# ========================
# TRANSFER PREPROCESS
# ========================

def preprocess_transfer(path, label, base_dir):

    path = tf.strings.join([base_dir, "/", path])
    img = _decode_image(path)

    img = tf.image.grayscale_to_rgb(img)
    img = efficientnet_preprocess(img)

    return img, label


def build_dataset(df, split, base_dir,
                  batch_size=32,
                  mode="baseline"):

    df_split = df[df["split"] == split]

    paths = df_split["image_path"].values
    labels = df_split["label"].values

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if split == "train":
        ds = ds.shuffle(2048)

    if mode == "baseline":
        ds = ds.map(lambda p, l: preprocess_baseline(p, l, base_dir),
                    num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.map(lambda p, l: preprocess_transfer(p, l, base_dir),
                    num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return ds
