from pathlib import Path
import sys
import numpy as np
import pandas as pd
import tensorflow as tf


IMG_SIZE = 224
BATCH_SIZE = 32


# Localiza a raiz do projeto
def encontrar_repo_root() -> Path:
    here = Path(__file__).resolve()
    candidatos = [
        here.parent,
        here.parent.parent,
        here.parent.parent.parent,
    ]
    for base in candidatos:
        if (base / "BestModel").exists() and (base / "data").exists():
            return base

    cur = here
    for _ in range(6):
        if (cur / "BestModel").exists() and (cur / "data").exists():
            return cur
        cur = cur.parent

    raise RuntimeError("Raiz do projeto não encontrada.")


# Configura todos os caminhos necessários
def configurar_caminhos():
    repo_root = encontrar_repo_root()
    model_path = repo_root / "BestModel" / "best_model.keras"
    test_csv_path = repo_root / "data" / "ligia-compviz" / "test.csv"
    test_images_dir = repo_root / "data" / "ligia-compviz" / "test_images" / "test_images"
    submission_dir = repo_root / "Submission"
    submission_path = submission_dir / "submission_membros.csv"

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    submission_dir.mkdir(parents=True, exist_ok=True)

    assert model_path.exists(), f"Modelo não encontrado: {model_path}"
    assert test_csv_path.exists(), f"test.csv não encontrado: {test_csv_path}"
    assert test_images_dir.exists(), f"Diretório de imagens não encontrado: {test_images_dir}"

    return model_path, test_csv_path, test_images_dir, submission_path


# Carrega o modelo .keras
def carregar_modelo(model_path: Path) -> tf.keras.Model:
    return tf.keras.models.load_model(model_path)


# Carrega o test.csv
def carregar_test_csv(test_csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(test_csv_path)
    if "id" not in df.columns:
        raise RuntimeError("test.csv deve conter a coluna 'id'.")
    return df


# Monta o caminho absoluto de cada imagem
def montar_caminhos_imagens(df: pd.DataFrame, test_images_dir: Path) -> pd.DataFrame:
    df = df.copy()
    df["filepath"] = df["id"].apply(lambda x: str(test_images_dir / x))

    exemplo = df.iloc[0]["filepath"]
    if not Path(exemplo).exists():
        raise RuntimeError(f"Imagem não encontrada: {exemplo}")

    return df


# Pré-processa cada imagem
def preprocess_image(path: tf.Tensor) -> tf.Tensor:
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize_with_pad(img, IMG_SIZE, IMG_SIZE)
    img = tf.cast(img, tf.float32)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img


# Cria o dataset de inferência
def criar_dataset_inferencia(df: pd.DataFrame) -> tf.data.Dataset:
    paths = tf.constant(df["filepath"].values, dtype=tf.string)
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


# Executa a inferência
def rodar_inferencia(model: tf.keras.Model, ds_test: tf.data.Dataset) -> np.ndarray:
    preds = model.predict(ds_test, verbose=1)
    if preds.ndim == 2 and preds.shape[1] == 1:
        preds = preds[:, 0]
    return preds.astype(float)


# Monta o dataframe final de submissão
def montar_submission(df: pd.DataFrame, preds: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({
        "id": df["id"].values,
        "target": preds,
    })


# Salva o arquivo de submissão
def salvar_submission(submission: pd.DataFrame, submission_path: Path) -> None:
    submission.to_csv(submission_path, index=False)
    print(f"Arquivo gerado: {submission_path}")


# Executa todo o fluxo
def main():
    model_path, test_csv_path, test_images_dir, submission_path = configurar_caminhos()

    model = carregar_modelo(model_path)
    df_test = carregar_test_csv(test_csv_path)
    df_test = montar_caminhos_imagens(df_test, test_images_dir)

    ds_test = criar_dataset_inferencia(df_test)
    preds = rodar_inferencia(model, ds_test)

    submission = montar_submission(df_test, preds)
    salvar_submission(submission, submission_path)


if __name__ == "__main__":
    main()