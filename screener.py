"""Автоматический сбор скриншотов игры.

Запуск:
    python screener.py --window "Кликер" --interval 3

Скрипт ищет окно браузера по части заголовка, делает скриншоты только этого
окна и пропускает почти одинаковые кадры (чтобы датасет не состоял из копий).

Остановка: Ctrl+C.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyautogui
import pygetwindow as gw

ROOT = Path(__file__).resolve().parent

# Совпадение координат окна и пикселей скриншота при масштабировании Windows.
try:
    import ctypes

    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    pass


def find_window(title_part: str):
    for window in gw.getAllWindows():
        title = (window.title or "").strip()
        if title and title_part.lower() in title.lower() and window.width > 200:
            return window
    return None


def grab(region):
    image = pyautogui.screenshot(region=region)
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", default="Кликер", help="часть заголовка окна")
    parser.add_argument("--interval", type=float, default=3.0)
    parser.add_argument("--diff", type=float, default=3.0, help="порог отличия кадров")
    parser.add_argument("--out", default="screenshots")
    parser.add_argument("--max", type=int, default=0, help="0 = без ограничения")
    args = parser.parse_args()

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    window = find_window(args.window)
    if window is None:
        print(f"Окно с заголовком «{args.window}» не найдено.")
        print("Открой игру в браузере или укажи другое слово через --window.")
        return

    region = (window.left, window.top, window.width, window.height)
    print(f"Окно: {window.title!r}")
    print(f"Область: {region}")
    print(f"Интервал: {args.interval} c. Остановка: Ctrl+C.")

    last_gray = None
    saved = 0
    try:
        while True:
            frame = grab(region)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (160, 90))
            if last_gray is not None:
                diff = float(np.mean(cv2.absdiff(gray, last_gray)))
                if diff < args.diff:
                    time.sleep(args.interval)
                    continue
            last_gray = gray
            name = f"screenshot_{datetime.now():%Y-%m-%d_%H%M%S}.png"
            cv2.imwrite(str(out / name), frame)
            saved += 1
            print(f"[{saved}] {name}")
            if args.max and saved >= args.max:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print(f"\nОстановлено. Сохранено: {saved}")


if __name__ == "__main__":
    main()
