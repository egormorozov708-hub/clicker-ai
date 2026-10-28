"""Сборка итогового датасета из всех размеченных скриншотов.

Классы:
  0 game        — персонаж текущего уровня (цель для кликов);
  1 improvement — кнопки покупки улучшений внизу слева;
  2 advertising — крестик закрытия рекламного окна;
  3 bonus       — бонусные кнопки 2X / +1K вверху слева.

Источники:
  screenshots/  — 84 старых кадра: game из JSON/Game_boxes.json, UI через HSV;
  level 2/      — 30 кадров второго уровня: ручные game и improvement;
  advertising/  — 19 кадров с рекламой: ручной advertising (крестик).

Запуск:
    python build_dataset.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
OLD_SRC = ROOT / "screenshots"
LEVEL2_SRC = ROOT / "level 2"
AD_SRC = ROOT / "advertising"
OUT = ROOT / "dataset_final"
OVERRIDES = ROOT / "game_boxes.json"
CLASSES = ["game", "improvement", "advertising", "bonus"]
CLASS_ID = {name: i for i, name in enumerate(CLASSES)}


def find_teal(img: np.ndarray) -> list[list[int]]:
    """Крупные бирюзовые кнопки интерфейса (2X, +1K, улучшения)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (85, 180, 50), (100, 255, 150))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 60 or h < 30 or w * h < 2500:
            continue
        if not (0.5 < w / h < 4.5):
            continue
        boxes.append([x, y, x + w, y + h])
    return boxes


def read_json_boxes(path: Path) -> list[tuple[str, list[float]]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    boxes = []
    for shape in data.get("shapes", []):
        if shape.get("shape_type") != "rectangle":
            continue
        xs = [float(p[0]) for p in shape["points"]]
        ys = [float(p[1]) for p in shape["points"]]
        boxes.append((shape["label"], [min(xs), min(ys), max(xs), max(ys)]))
    return boxes


def read_old_game(img_path: Path, overrides: dict) -> list[float] | None:
    if img_path.name in overrides:
        return [float(v) for v in overrides[img_path.name]]
    for label, box in read_json_boxes(img_path.with_suffix(".json")):
        if label == "game":
            return box
    return None


def clip_box(box: list[float], w: int, h: int) -> list[float] | None:
    x1, y1, x2, y2 = box
    x1, x2 = max(0.0, x1), min(float(w), x2)
    y1, y2 = max(0.0, y1), min(float(h), y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return [x1, y1, x2, y2]


def yolo_line(cls: int, box: list[float], w: int, h: int) -> str:
    x1, y1, x2, y2 = box
    return (
        f"{cls} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
        f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}\n"
    )


def old_items(overrides: dict) -> list[tuple[Path, list[tuple[int, list[float]]]]]:
    """Старые 84 кадра: game вручную/авто, UI — цветовая сегментация."""
    items = []
    for img_path in sorted(OLD_SRC.glob("*.png")):
        game = read_old_game(img_path, overrides)
        if not game:
            print(f"[skip] {img_path.name}: нет рамки game")
            continue
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        h, w = image.shape[:2]
        labels: list[tuple[int, list[float]]] = [(CLASS_ID["game"], game)]

        teal = find_teal(image)
        bonus = [b for b in teal if b[2] < w * 0.45 and b[3] < h * 0.45]
        improvement = [b for b in teal if b[2] < w * 0.45 and b[1] > h * 0.55]
        bonus = sorted(bonus, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)[:2]
        improvement = sorted(
            improvement, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True
        )[:2]
        labels += [(CLASS_ID["bonus"], b) for b in bonus]
        labels += [(CLASS_ID["improvement"], b) for b in improvement]
        items.append((img_path, labels))
    return items


def level2_items() -> list[tuple[Path, list[tuple[int, list[float]]]]]:
    """Второй уровень: ручные game + improvement из JSON."""
    items = []
    for img_path in sorted(LEVEL2_SRC.glob("*.png")):
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        h, w = image.shape[:2]
        labels: list[tuple[int, list[float]]] = []
        for label, box in read_json_boxes(img_path.with_suffix(".json")):
            if label == "game":
                bw, bh = box[2] - box[0], box[3] - box[1]
                # На трёх кадрах вместо персонажа обведён текст «Уровень 2».
                if bh < 70 or bw / max(bh, 1.0) > 1.5:
                    print(f"[skip box] {img_path.name}: подозрительная game-рамка {bw:.0f}x{bh:.0f}")
                    continue
                clipped = clip_box(box, w, h)
                if clipped:
                    labels.append((CLASS_ID["game"], clipped))
            elif label == "improvement":
                clipped = clip_box(box, w, h)
                if clipped:
                    labels.append((CLASS_ID["improvement"], clipped))
        if labels:
            items.append((img_path, labels))
    return items


def advertising_items() -> list[tuple[Path, list[tuple[int, list[float]]]]]:
    """Реклама: ручной крестик закрытия."""
    items = []
    for img_path in sorted(AD_SRC.glob("*.png")):
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        h, w = image.shape[:2]
        labels: list[tuple[int, list[float]]] = []
        for label, box in read_json_boxes(img_path.with_suffix(".json")):
            if label != "advertising":
                continue
            clipped = clip_box(box, w, h)
            if clipped:
                labels.append((CLASS_ID["advertising"], clipped))
        if labels:
            items.append((img_path, labels))
    return items


def split_items(items):
    """Каждый пятый кадр источника — в val; так все классы попадают в обе части."""
    val = set(range(4, len(items), 5))
    return [(item, i in val) for i, item in enumerate(items)]


def main() -> None:
    overrides = {}
    if OVERRIDES.exists():
        overrides = json.loads(OVERRIDES.read_text(encoding="utf-8"))

    groups = [
        ("old", split_items(old_items(overrides))),
        ("level2", split_items(level2_items())),
        ("advertising", split_items(advertising_items())),
    ]

    if OUT.exists():
        shutil.rmtree(OUT)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    counts = {name: 0 for name in CLASSES}
    total = 0
    val_total = 0
    for group_name, entries in groups:
        group_counts = {name: 0 for name in CLASSES}
        group_val = 0
        for (img_path, labels), is_val in entries:
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            h, w = image.shape[:2]
            split = "val" if is_val else "train"
            out_name = img_path.stem + ".jpg"
            cv2.imwrite(
                str(OUT / "images" / split / out_name),
                image,
                [cv2.IMWRITE_JPEG_QUALITY, 92],
            )
            text = ""
            for cls, box in labels:
                clipped = clip_box(box, w, h)
                if clipped:
                    text += yolo_line(cls, clipped, w, h)
                    counts[CLASSES[cls]] += 1
                    group_counts[CLASSES[cls]] += 1
            (OUT / "labels" / split / (img_path.stem + ".txt")).write_text(
                text, encoding="utf-8"
            )
            total += 1
            if is_val:
                val_total += 1
                group_val += 1
        print(f"{group_name}: кадров {len(entries)}, val {group_val}, объектов {group_counts}")

    (ROOT / "data_final.yaml").write_text(
        "path: ./dataset_final\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 4\n"
        "names: ['game', 'improvement', 'advertising', 'bonus']\n",
        encoding="utf-8",
    )
    print(f"Готово: {total} картинок, train={total - val_total}, val={val_total}")
    print("Объектов:", counts)
    print(f"Датасет: {OUT}")


if __name__ == "__main__":
    main()
