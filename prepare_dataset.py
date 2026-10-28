"""Подготовка датасета из разметки X-AnyLabeling (JSON) в формат YOLO.

Использование:
    python prepare_dataset.py

Что делает:
1. Берёт все PNG из папки screenshots, для которых есть одноимённый .json.
2. Преобразует прямоугольники из JSON в YOLO-формат (class x_center y_center w h).
3. При необходимости уточняет рамку персонажа с помощью предобученной YOLOv8n
   (класс person). Это даёт более полные рамки, чем ручная разметка по торсу.
4. Делит данные на train/val (80/20) и создаёт data_v2.yaml.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "screenshots"
OUT = ROOT / "dataset_v2"
CLASS_NAME = "game"
REFINE_WITH_PERSON = True


def box_from_json(path: Path) -> list[float] | None:
    data = json.loads(path.read_text(encoding="utf-8"))
    shapes = [s for s in data.get("shapes", []) if s.get("label") == CLASS_NAME]
    if not shapes:
        return None
    pts = shapes[0]["points"]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return [min(xs), min(ys), max(xs), max(ys)]


def iou(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua else 0.0


def refine_box(img_path: Path, old: list[float], model: YOLO) -> list[float]:
    """Ищем человека рядом с ручной рамкой и возвращаем более полную рамку."""
    result = model(str(img_path), imgsz=1280, conf=0.15, verbose=False)[0]
    best = None
    for b in result.boxes:
        if int(b.cls) != 0:  # person
            continue
        box = [float(v) for v in b.xyxy[0]]
        score = iou(box, old)
        if score >= 0.45 and (best is None or score > best[0]):
            best = (score, box)
    return best[1] if best else old


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    person = YOLO("yolov8n.pt") if REFINE_WITH_PERSON else None
    pairs: list[tuple[Path, list[float]]] = []
    for img in sorted(SRC.glob("*.png")):
        label = img.with_suffix(".json")
        if not label.exists():
            continue
        box = box_from_json(label)
        if box is None:
            continue
        if person:
            box = refine_box(img, box, person)
        pairs.append((img, box))

    if not pairs:
        raise SystemExit("Не найдено ни одной размеченной картинки в screenshots/")

    # 80/20: каждая пятая картинка уходит в val
    val_idx = set(range(4, len(pairs), 5))
    for i, (img, box) in enumerate(pairs):
        split = "val" if i in val_idx else "train"
        out_img = OUT / "images" / split / (img.stem + ".jpg")
        image = cv2.imread(str(img))
        cv2.imwrite(str(out_img), image, [cv2.IMWRITE_JPEG_QUALITY, 92])

        h, w = image.shape[:2]
        x1, y1, x2, y2 = box
        xc, yc = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
        bw, bh = (x2 - x1) / w, (y2 - y1) / h
        txt = OUT / "labels" / split / (img.stem + ".txt")
        txt.write_text(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n", encoding="utf-8")

    (ROOT / "data_v2.yaml").write_text(
        "path: ./dataset_v2\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 1\n"
        "names: ['game']\n",
        encoding="utf-8",
    )
    n_train = len(pairs) - len(val_idx)
    print(f"Готово: {len(pairs)} картинок, train={n_train}, val={len(val_idx)}")
    print(f"Датасет: {OUT}")
    print(f"Конфиг: {ROOT / 'data_v2.yaml'}")


if __name__ == "__main__":
    main()
