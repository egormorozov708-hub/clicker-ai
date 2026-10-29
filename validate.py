"""Проверка модели на папке со скриншотами и сохранение примеров детекции.

Запуск:
    python validate.py --model runs/detect/clicker_levels/weights/best.pt --source "level 2" --limit 8
    python validate.py --model runs/detect/clicker_levels/weights/best.pt --data data_final.yaml

Первый режим рисует найденные объекты и складывает картинки в docs/examples.
Второй считает метрики mAP на отложенной выборке val.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO

COLORS = {
    "game": (0, 255, 60),
    "improvement": (255, 200, 0),
    "advertising": (255, 0, 255),
    "bonus": (0, 255, 255),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="runs/detect/clicker_levels/weights/best.pt"
    )
    parser.add_argument("--source", default="screenshots")
    parser.add_argument("--data", default="")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--limit", type=int, default=0, help="0 = все изображения")
    parser.add_argument("--out", default="docs/examples")
    parser.add_argument("--metrics-out", default="docs/metrics.json")
    args = parser.parse_args()

    model = YOLO(args.model)
    print("Классы модели:", model.names)

    if args.data:
        metrics = model.val(data=args.data, imgsz=args.imgsz, workers=0)
        print(f"mAP50:    {metrics.box.map50:.4f}")
        print(f"mAP50-95: {metrics.box.map:.4f}")
        per_class = []
        for i, name in model.names.items():
            row = {
                "class": name,
                "precision": round(float(metrics.box.p[i]), 4),
                "recall": round(float(metrics.box.r[i]), 4),
                "map50": round(float(metrics.box.ap50[i]), 4),
                "map50_95": round(float(metrics.box.ap[i]), 4),
            }
            per_class.append(row)
            print(
                f"  {name:12} P={row['precision']:.3f} "
                f"R={row['recall']:.3f} mAP50={row['map50']:.3f} "
                f"mAP50-95={row['map50_95']:.3f}"
            )
        summary = {
            "overall": {
                "precision": round(float(metrics.box.mp), 4),
                "recall": round(float(metrics.box.mr), 4),
                "map50": round(float(metrics.box.map50), 4),
                "map50_95": round(float(metrics.box.map), 4),
            },
            "per_class": per_class,
        }
        out = Path(args.metrics_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Метрики сохранены: {out}")
        return

    source = Path(args.source)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    files = [
        p for p in sorted(source.glob("*"))
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    ]
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"В папке {source} нет изображений")

    totals: Counter[str] = Counter()
    saved = 0
    for path in files:
        image = cv2.imread(str(path))
        if image is None:
            continue
        result = model(str(path), imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
        for box in result.boxes:
            name = model.names[int(box.cls)]
            totals[name] += 1
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            color = COLORS.get(name, (200, 200, 200))
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
            cv2.putText(
                image,
                f"{name} {float(box.conf):.2f}",
                (x1, max(24, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
                cv2.LINE_AA,
            )
        dst = out / path.name
        cv2.imwrite(str(dst), image)
        saved += 1
        print(f"{path.name}: " + ", ".join(
            f"{model.names[int(b.cls)]} {float(b.conf):.2f}" for b in result.boxes
        ))
    print(f"\nСохранено примеров: {saved} -> {out}")
    print("Найдено объектов:", dict(totals))


if __name__ == "__main__":
    main()
