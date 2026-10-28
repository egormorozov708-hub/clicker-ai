"""Сборка итогового датасета из скриншотов, ручной разметки и авторазметки UI.

Классы:
  0 game        — персонаж (цель для кликов)
  1 upgrade     — кнопки улучшений внизу слева
  2 advertising — кнопка отключения рекламы справа сверху
  3 bonus       — бонусные кнопки (2X, +100) вверху слева

Запуск:
    python build_dataset.py

Источники рамок game:
  * screenshots/*.json  — ручная разметка X-AnyLabeling;
  * game_boxes.json      — авто/ручная разметка остальных картинок
                           (словарь "имя_файла.png": [x1, y1, x2, y2]).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "screenshots"
OUT = ROOT / "dataset_final"
OVERRIDES = ROOT / "game_boxes.json"
CLASSES = ["game", "upgrade", "advertising", "bonus"]


def find_teal(img: np.ndarray) -> list[list[int]]:
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


def find_purple(img: np.ndarray) -> list[list[int]]:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (115, 100, 120), (140, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 180 or h < 20 or w / h < 3:
            continue
        boxes.append([x, y, x + w, y + h])
    return boxes


def read_game_box(img_path: Path, overrides: dict) -> list[float] | None:
    if img_path.name in overrides:
        return [float(v) for v in overrides[img_path.name]]
    label = img_path.with_suffix(".json")
    if not label.exists():
        return None
    data = json.loads(label.read_text(encoding="utf-8"))
    shapes = [s for s in data.get("shapes", []) if s.get("label") == "game"]
    if not shapes:
        return None
    pts = shapes[0]["points"]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return [min(xs), min(ys), max(xs), max(ys)]


def yolo_line(cls: int, box: list[float], w: int, h: int) -> str:
    x1, y1, x2, y2 = box
    x1, x2 = max(0.0, x1), min(float(w), x2)
    y1, y2 = max(0.0, y1), min(float(h), y2)
    if x2 <= x1 or y2 <= y1:
        return ""
    return (
        f"{cls} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
        f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}\n"
    )


def main() -> None:
    overrides = {}
    if OVERRIDES.exists():
        overrides = json.loads(OVERRIDES.read_text(encoding="utf-8"))

    if OUT.exists():
        shutil.rmtree(OUT)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    items = []
    counts = {name: 0 for name in CLASSES}
    for img_path in sorted(SRC.glob("*.png")):
        game = read_game_box(img_path, overrides)
        if not game:
            print(f"[skip] {img_path.name}: нет рамки game")
            continue
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        h, w = image.shape[:2]

        labels: list[tuple[int, list[float]]] = [(0, game)]
        counts["game"] += 1

        teal = find_teal(image)
        bonus = [b for b in teal if b[2] < w * 0.45 and b[3] < h * 0.45]
        upgrade = [b for b in teal if b[2] < w * 0.45 and b[1] > h * 0.55]
        # самые крупные/нижние кнопки, чтобы не брать мелкий текст
        bonus = sorted(bonus, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)[:2]
        upgrade = sorted(upgrade, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)[:2]
        for box in bonus:
            labels.append((3, box))
            counts["bonus"] += 1
        for box in upgrade:
            labels.append((1, box))
            counts["upgrade"] += 1

        advertising = [
            b for b in find_purple(image)
            if b[0] > w * 0.45 and b[3] < h * 0.4 and (b[2] - b[0]) > 180
        ]
        for box in advertising[:1]:
            labels.append((2, box))
            counts["advertising"] += 1

        items.append((img_path, image, labels))

    if not items:
        raise SystemExit("Нет данных для сборки датасета")

    val_idx = set(range(4, len(items), 5))
    for i, (img_path, image, labels) in enumerate(items):
        split = "val" if i in val_idx else "train"
        cv2.imwrite(
            str(OUT / "images" / split / (img_path.stem + ".jpg")),
            image,
            [cv2.IMWRITE_JPEG_QUALITY, 92],
        )
        h, w = image.shape[:2]
        text = "".join(yolo_line(cls, box, w, h) for cls, box in labels)
        (OUT / "labels" / split / (img_path.stem + ".txt")).write_text(text, encoding="utf-8")

    (ROOT / "data_final.yaml").write_text(
        "path: ./dataset_final\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 4\n"
        "names: ['game', 'upgrade', 'advertising', 'bonus']\n",
        encoding="utf-8",
    )
    print(f"Готово: {len(items)} картинок, train={len(items) - len(val_idx)}, val={len(val_idx)}")
    print("Объектов:", counts)
    print(f"Датасет: {OUT}")


if __name__ == "__main__":
    main()
