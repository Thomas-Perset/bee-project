from pathlib import Path
import re
import cv2
import numpy as np
import pandas as pd

ID_RE = re.compile(r'(\d+)')

def id_from_path(p: Path) -> int:
    m = ID_RE.search(p.stem)
    if not m:
        raise ValueError(f"ID introuvable dans {p.name}")
    return int(m.group(1))

def load_image(path: Path):
    img = cv2.imread(str(path))[:, :, ::-1]  # BGR->RGB
    return img

def load_mask(path: Path):
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    # binaire 0/1
    return (m > 0).astype(np.uint8)

def scan_pairs(images_dir: Path, masks_dir: Path):
    imgs = sorted(Path(images_dir).glob('*'))
    masks = sorted(Path(masks_dir).glob('*'))
    by_id_img = {id_from_path(p): p for p in imgs}
    by_id_msk = {id_from_path(p): p for p in masks}
    common = sorted(set(by_id_img) & set(by_id_msk))
    return [(i, by_id_img[i], by_id_msk[i]) for i in common]

def read_labels(labels_xlsx: Path):
    df = pd.read_excel(labels_xlsx)
    # normaliser noms de colonnes
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    # Assumer colonne id -> 'id' ; types -> 'bug_type', 'species'
    return df
