from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def extract_patient_id(filename: str) -> str:
    base = filename.rsplit(".", 1)[0]
    parts = base.split("-")

    # esperado: CLASS-PATIENT-INDEX  (ex: NORMAL-3514363-0002)
    if len(parts) >= 3 and parts[1].isdigit():
        return parts[1]

    # fallback: tenta achar um bloco numérico grande no nome
    nums = [p for p in parts if p.isdigit()]
    if nums:
        return max(nums, key=len)

    return "unknown"


def _label_map_from_dirs(train_path: Path) -> dict:
    classes = sorted([d.name for d in train_path.iterdir() if d.is_dir()])
    if not classes:
        raise FileNotFoundError(f"Sem classes em: {train_path}")
    return {c: i for i, c in enumerate(classes)}


def build_dataframe(base_path: Path, project_root: Path, class_to_label: dict) -> pd.DataFrame:
    rows = []
    if not base_path.exists():
        return pd.DataFrame(columns=["image_path", "filename", "class", "label", "patient_id"])

    for class_dir in sorted(base_path.iterdir()):
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        if class_name not in class_to_label:
            raise ValueError(f"Classe inesperada: {class_name}")

        for img_path in sorted(class_dir.iterdir()):
            if not img_path.is_file():
                continue

            filename = img_path.name
            rel_path = img_path.resolve().relative_to(project_root).as_posix()
            pid = extract_patient_id(filename)

            rows.append(
                {
                    "image_path": rel_path,
                    "filename": filename,
                    "class": class_name,
                    "label": class_to_label[class_name],
                    "patient_id": pid,
                }
            )

    return pd.DataFrame(rows)


def _validate_patient_ids(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("DataFrame vazio. Verifique se train/test têm imagens nas pastas de classe.")

    unknown_ratio = (df["patient_id"] == "unknown").mean()
    if unknown_ratio > 0.02:
        raise ValueError(
            f"Muitos patient_id ficaram 'unknown' ({unknown_ratio:.2%}). "
            "Isso indica que a extração do patient_id não está batendo com o padrão do filename."
        )

    # checagem simples: para cada patient_id, deve haver pelo menos 1 imagem
    if df["patient_id"].nunique() < 10:
        raise ValueError(
            "Poucos patient_id únicos detectados. Possível extração incorreta (muitos arquivos caindo no mesmo id)."
        )


def group_split(df: pd.DataFrame, val_size: float = 0.2, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    train_idx, val_idx = next(splitter.split(df, groups=df["patient_id"]))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    return train_df, val_df


def _assert_no_patient_overlap(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    train_ids = set(train_df["patient_id"])
    val_ids = set(val_df["patient_id"])
    test_ids = set(test_df["patient_id"])

    assert len(train_ids & val_ids) == 0, "Vazamento: patient_id repetido entre train e val."
    assert len(train_ids & test_ids) == 0, "Vazamento: patient_id repetido entre train e test."
    assert len(val_ids & test_ids) == 0, "Vazamento: patient_id repetido entre val e test."


def prepare_data(data_dir: str = "data/chest_xray", val_size: float = 0.2, seed: int = 42):
    root = _project_root()
    data_path = (root / data_dir).resolve()

    train_path = data_path / "train"
    test_path = data_path / "test"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Esperado:\n- {train_path}\n- {test_path}")

    class_to_label = _label_map_from_dirs(train_path)

    train_df_full = build_dataframe(train_path, project_root=root, class_to_label=class_to_label)
    test_df = build_dataframe(test_path, project_root=root, class_to_label=class_to_label)

    _validate_patient_ids(train_df_full)
    _validate_patient_ids(test_df)

    train_df, val_df = group_split(train_df_full, val_size=val_size, seed=seed)

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    _assert_no_patient_overlap(train_df, val_df, test_df)

    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    csv_path = data_path / "dataset.csv"
    full_df.to_csv(csv_path, index=False)

    return train_df, val_df, test_df, csv_path, class_to_label
