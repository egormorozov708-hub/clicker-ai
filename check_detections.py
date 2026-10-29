"""Проверка детекций итоговой модели на датасете (практические метрики).

Запуск:
    python check_detections.py --model runs/detect/clicker_levels/weights/best.pt --conf 0.35

Скрипт сравнивает предсказания с ground truth по IoU и печатает:
  * сколько кадров содержат корректную рамку game;
  * средний IoU лучшей рамки game;
  * ложные срабатывания по классам (рамка без пары);
  * сколько рамок game прошло бы фильтр app.py.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
NAMES = {0: "game", 1: "improvement", 2: "advertising", 3: "bonus"}


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
    if bw < 12 or bh < 35 or bh / max(bw, 1.0) < 0.65:
        return False
    if (x1 + x2) / 2 > w * 0.97:
        return False
    return True


def app_ad_box_ok(box, w: int, h: int) -> bool:
    """Тот же фильтр, что в app.py для крестика рекламы."""
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    if not (40 <= bw <= 200 and 40 <= bh <= 200):
        return False
    if not (0.6 <= bw / max(bh, 1.0) <= 2.2):
        return False
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    if cx < w * 0.45 or cy > h * 0.85:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="runs/detect/clicker_levels/weights/best.pt"
    )
    parser.add_argument("--images", default="dataset_final/images/val")
    parser.add_argument("--labels", default="dataset_final/labels/val")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--json-out", default="docs/check.json")
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
    game_frames = 0
    ad_found = 0
    ad_passed_filter = 0
    ad_frames = 0

    for img_path in images:
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        h, w = image.shape[:2]
        gt = read_gt(labels / f"{img_path.stem}.txt", w, h)
        game_frames += int(any(gcls == 0 for gcls, _ in gt))
        ad_frames += int(any(gcls == 2 for gcls, _ in gt))
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
                elif cls == 2:
                    ad_found += 1
                    if app_ad_box_ok(box, w, h):
                        ad_passed_filter += 1
            else:
                fp[NAMES[cls]] += 1
        for i, (gcls, _) in enumerate(gt):
            if i not in matched_gt:
                fn[NAMES[gcls]] += 1

    n = len(images)
    print(f"Кадров: {n}")
    print(f"game найдена (IoU>=0.5): {game_found}/{game_frames} = {game_found / max(game_frames, 1):.1%}")
    if game_iou:
        print(f"Средний IoU лучшей game-рамки: {sum(game_iou) / len(game_iou):.3f}")
    print(f"game проходит фильтр app.py: {passed_filter}/{game_found}")
    print(f"advertising проходит фильтр app.py: {ad_passed_filter}/{ad_found}")
    print("\nКласс         TP   FP   FN")
    for name in NAMES.values():
        print(f"{name:12} {tp[name]:4} {fp[name]:4} {fn[name]:4}")

    if args.json_out:
        summary = {
            "frames": n,
            "conf": args.conf,
            "game_frames": game_frames,
            "game_found": game_found,
            "game_iou": round(sum(game_iou) / len(game_iou), 3) if game_iou else 0.0,
            "game_passed_filter": passed_filter,
            "ad_frames": ad_frames,
            "ad_found": ad_found,
            "ad_passed_filter": ad_passed_filter,
            "tp": dict(tp),
            "fp": dict(fp),
            "fn": dict(fn),
        }
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"JSON сохранён: {out}")


if __name__ == "__main__":
    main()
