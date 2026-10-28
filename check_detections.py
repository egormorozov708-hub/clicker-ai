"""Проверка детекций итоговой модели на датасете (практические метрики).

Запуск:
    python check_detections.py --model best.pt --conf 0.35

Скрипт сравнивает предсказания с ground truth по IoU и печатает:
  * сколько кадров содержат корректную рамку game;
  * средний IoU лучшей рамки game;
  * ложные срабатывания по классам (рамка без пары);
  * сколько рамок game прошло бы фильтр app.py.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
NAMES = {0: "game", 1: "upgrade", 2: "advertising", 3: "bonus"}


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua else 0.0


def read_gt(label_path: Path, w: int, h: int):
    items = []
    if not label_path.exists():
        return items
    for line in label_path.read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) != 5:
            continue
        cls = int(float(p[0]))
        xc, yc, bw, bh = map(float, p[1:])
        items.append((
            cls,
            [(xc - bw / 2) * w, (yc - bh / 2) * h,
             (xc + bw / 2) * w, (yc + bh / 2) * h],
        ))
    return items


def app_game_box_ok(box, w: int, h: int) -> bool:
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    if bw < 12 or bh < 35 or bh / max(bw, 1.0) < 1.05:
        return False
    if (x1 + x2) / 2 > w * 0.94:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="best.pt")
    parser.add_argument("--images", default="dataset_final/images/val")
    parser.add_argument("--labels", default="dataset_final/labels/val")
    parser.add_argument("--conf", type=float, default=0.35)
    args = parser.parse_args()

    model = YOLO(args.model)
    images = sorted(Path(args.images).glob("*.jpg")) + sorted(Path(args.images).glob("*.png"))
    labels = Path(args.labels)

    tp = Counter()
    fp = Counter()
    fn = Counter()
    game_found = 0
    game_iou = []
    passed_filter = 0

    for img_path in images:
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        h, w = image.shape[:2]
        gt = read_gt(labels / f"{img_path.stem}.txt", w, h)
        result = model(str(img_path), imgsz=640, conf=args.conf, verbose=False)[0]
        preds = [(int(b.cls), [float(v) for v in b.xyxy[0]], float(b.conf)) for b in result.boxes]

        matched_gt = set()
        for cls, box, conf in preds:
            best_iou, best_i = 0.0, None
            for i, (gcls, gbox) in enumerate(gt):
                if i in matched_gt or gcls != cls:
                    continue
                score = iou(box, gbox)
                if score > best_iou:
                    best_iou, best_i = score, i
            if best_iou >= 0.5 and best_i is not None:
                tp[NAMES[cls]] += 1
                matched_gt.add(best_i)
                if cls == 0:
                    game_found += 1
                    game_iou.append(best_iou)
                    if app_game_box_ok(box, w, h):
                        passed_filter += 1
            else:
                fp[NAMES[cls]] += 1
        for i, (gcls, _) in enumerate(gt):
            if i not in matched_gt:
                fn[NAMES[gcls]] += 1

    n = len(images)
    print(f"Кадров: {n}")
    print(f"game найдена (IoU>=0.5): {game_found}/{n} = {game_found / max(n, 1):.1%}")
    if game_iou:
        print(f"Средний IoU лучшей game-рамки: {sum(game_iou) / len(game_iou):.3f}")
    print(f"game проходит фильтр app.py: {passed_filter}/{game_found}")
    print("\nКласс         TP   FP   FN")
    for name in NAMES.values():
        print(f"{name:12} {tp[name]:4} {fp[name]:4} {fn[name]:4}")


if __name__ == "__main__":
    main()
