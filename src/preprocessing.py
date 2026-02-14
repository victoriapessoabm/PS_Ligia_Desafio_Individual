from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _data_paths(data_dir: str) -> Tuple[Path, Path, Path, Path]:
    root = _project_root()
    data_path = (root / data_dir).resolve()
    train_path = data_path / "train"
    test_path = data_path / "test"
    return root, data_path, train_path, test_path


def extract_patient_id(filename: str) -> str:
    base = filename.rsplit(".", 1)[0]
    parts = base.split("-")

    if len(parts) >= 3 and parts[1].isdigit():
        return parts[1]

    nums = [p for p in parts if p.isdigit()]
    return max(nums, key=len) if nums else "unknown"


def _label_map_from_dirs(train_path: Path) -> Dict[str, int]:
    classes = sorted([d.name for d in train_path.iterdir() if d.is_dir()])
    return {c: i for i, c in enumerate(classes)}


def _img_hash(path: Path) -> str:
    with Image.open(path) as im:
        return hashlib.md5(im.convert("L").tobytes()).hexdigest()


def build_dataframe(base_path: Path, data_root: Path,
                    class_to_label: Dict[str, int], split_source: str):

    rows = []

    for class_dir in sorted(base_path.iterdir()):
        if not class_dir.is_dir():
            continue

        for img_path in sorted(class_dir.iterdir()):
            if not img_path.is_file():
                continue

            rows.append({
                "image_path": img_path.relative_to(data_root).as_posix(),
                "filename": img_path.name,
                "class": class_dir.name,
                "label": class_to_label[class_dir.name],
                "patient_id": extract_patient_id(img_path.name),
                "split_source": split_source,
                "img_hash": _img_hash(img_path)
            })

    return pd.DataFrame(rows)


def stratified_split_by_patient(df, val_size, seed):

    per_patient = df.groupby("patient_id").agg(
        patient_label=("label", lambda s: int(s.mode().iloc[0]))
    ).reset_index()

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_size,
        random_state=seed
    )

    train_idx, val_idx = next(
        splitter.split(per_patient["patient_id"], per_patient["patient_label"])
    )

    train_ids = set(per_patient.iloc[train_idx]["patient_id"])
    val_ids = set(per_patient.iloc[val_idx]["patient_id"])

    train_df = df[df["patient_id"].isin(train_ids)].reset_index(drop=True)
    val_df = df[df["patient_id"].isin(val_ids)].reset_index(drop=True)

    return train_df, val_df


def prepare_data(
    data_dir="data/chest_xray",
    val_size=0.2,
    seed=42
):

    root, data_path, train_path, test_path = _data_paths(data_dir)

    class_to_label = _label_map_from_dirs(train_path)

    train_df = build_dataframe(train_path, data_path, class_to_label, "train")
    test_df = build_dataframe(test_path, data_path, class_to_label, "test")

    train_df, val_df = stratified_split_by_patient(train_df, val_size, seed)

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    csv_path = data_path / "dataset.csv"
    full_df.to_csv(csv_path, index=False)

    return full_df, csv_path
