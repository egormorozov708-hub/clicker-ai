"""Clicker AI — бот для игры «Кликер: Становление шкафа».

Что умеет:
  * находит персонажа (класс game) и кликает по нему;
  * покупает улучшения (класс upgrade);
  * нажимает бонусные кнопки 2X / +100 (класс bonus);
  * при желании нажимает кнопку отключения рекламы (класс advertising);
  * показывает превью с рамками (--preview) или работает без окна.

Запуск:
    python app.py --model best.pt
    python app.py --model best.pt --preview --dry-run
    python app.py --model best.pt --window "Кликер" --crop-window

Горячие клавиши (если установлен pynput):
    F8  — пауза / продолжить
    F9  — выход
    Esc — выход
Без pynput: Ctrl+C в консоли.

Важно: бот кликает только когда активно окно игры (проверка по заголовку).
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pyautogui
import pygetwindow as gw
from ultralytics import YOLO

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.0

# На Windows при масштабировании интерфейса (125 %, 150 %) координаты pyautogui
# могут не совпадать с пикселями скриншота. Включаем DPI-awareness.
try:
    import ctypes

    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    pass

COLORS = {
    "game": (0, 255, 60),
    "upgrade": (255, 200, 0),
    "advertising": (255, 0, 255),
    "bonus": (0, 255, 255),
}


@dataclass
class Config:
    model: str = "best.pt"
    window: str = "Кликер"
    crop_window: bool = False
    preview: bool = False
    dry_run: bool = False
    imgsz: int = 640
    conf: float = 0.45
    click_interval: float = 0.08
    upgrade_interval: float = 3.0
    bonus_interval: float = 12.0
    ad_interval: float = 8.0
    click_game: bool = True
    click_upgrades: bool = True
    click_bonuses: bool = True
    close_ads: bool = False
    max_seconds: float = 0.0


@dataclass
class Stats:
    frames: int = 0
    clicks: int = 0
    started: float = field(default_factory=time.time)
    last_game_click: float = 0.0
    last_upgrade_click: float = 0.0
    last_bonus_click: float = 0.0
    last_ad_click: float = 0.0
    upgrade_idx: int = 0
    bonus_idx: int = 0

    @property
    def fps(self) -> float:
        elapsed = max(1e-6, time.time() - self.started)
        return self.frames / elapsed


def find_window(title_part: str):
    candidates = []
    for window in gw.getAllWindows():
        title = (window.title or "").strip()
        if title and title_part.lower() in title.lower() and window.width > 300:
            candidates.append(window)
    if not candidates:
        return None
    return max(candidates, key=lambda w: w.width * w.height)


def list_windows() -> None:
    print("Окна с заголовками:")
    for window in gw.getAllWindows():
        title = (window.title or "").strip()
        if title and window.width > 200:
            print(f"  {title!r}  ({window.left},{window.top},{window.width},{window.height})")


def grab_screen(region: tuple[int, int, int, int] | None) -> np.ndarray:
    image = pyautogui.screenshot(region=region)
    frame = np.array(image)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def center(box: list[float]) -> tuple[int, int]:
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def valid_game_box(box: list[float], frame_w: int, frame_h: int) -> bool:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if w < 12 or h < 35:
        return False
    if h / max(w, 1.0) < 1.05:
        return False
    if (x1 + x2) / 2 > frame_w * 0.94:  # правая колонка с рекламой
        return False
    return True


def detect(model: YOLO, frame: np.ndarray, conf: float, imgsz: int) -> list[tuple[str, float, list[float]]]:
    result = model(frame, imgsz=imgsz, conf=conf, verbose=False)[0]
    detections = []
    for box in result.boxes:
        name = model.names[int(box.cls)]
        detections.append((name, float(box.conf), [float(v) for v in box.xyxy[0]]))
    return detections


def draw(frame: np.ndarray, detections: list[tuple[str, float, list[float]]], stats: Stats, paused: bool) -> np.ndarray:
    canvas = frame.copy()
    for name, score, box in detections:
        color = COLORS.get(name, (200, 200, 200))
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            canvas, f"{name} {score:.2f}", (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
        )
    status = "PAUSED" if paused else "RUNNING"
    text = f"{status} | FPS {stats.fps:.1f} | clicks {stats.clicks}"
    cv2.putText(canvas, text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


class Preview:
    """Окно превью: OpenCV, а при сборке без GUI — Tkinter.

    Официальные wheel-пакеты OpenCV для новых версий Python иногда собираются
    без HighGUI (cv2.namedWindow падает с "The function is not implemented").
    Tkinter входит в стандартную библиотеку, поэтому используется как резерв.
    """

    def __init__(self, max_width: int = 1280, max_height: int = 720):
        self.max_width = max_width
        self.max_height = max_height
        self.backend = ""
        self.closed = False
        self._tk = None
        self._label = None
        self._photo = None
        try:
            cv2.namedWindow("Clicker AI", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Clicker AI", max_width, max_height)
            self.backend = "opencv"
            return
        except cv2.error:
            pass

        try:
            import tkinter as tk
            from PIL import Image, ImageTk
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Не удалось открыть окно превью: OpenCV собран без GUI, "
                "а Tkinter/Pillow недоступны. Запустите без --preview."
            ) from exc

        self._tk = tk
        self._Image = Image
        self._ImageTk = ImageTk
        self._root = tk.Tk()
        self._root.title("Clicker AI")
        self._root.protocol("WM_DELETE_WINDOW", self.close)
        self._root.bind("<Escape>", lambda _event: self.close())
        self._root.bind("q", lambda _event: self.close())
        self._label = tk.Label(self._root, bg="black")
        self._label.pack(fill="both", expand=True)
        self.backend = "tkinter"
        print("Превью: OpenCV без GUI, использую окно Tkinter.")

    def show(self, frame: np.ndarray) -> int:
        """Показать кадр. Возвращает код клавиши как cv2.waitKey (255 = нет)."""
        if self.closed:
            return -1
        if self.backend == "opencv":
            cv2.imshow("Clicker AI", frame)
            return cv2.waitKey(1) & 0xFF

        height, width = frame.shape[:2]
        scale = min(self.max_width / width, self.max_height / height, 1.0)
        if scale < 1.0:
            frame = cv2.resize(
                frame, (int(width * scale), int(height * scale)),
                interpolation=cv2.INTER_AREA,
            )
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = self._Image.fromarray(rgb)
        self._photo = self._ImageTk.PhotoImage(image)
        self._label.configure(image=self._photo)
        self._root.update_idletasks()
        self._root.update()
        return -1 if self.closed else 255

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.backend == "tkinter":
            try:
                self._root.destroy()
            except Exception:
                pass
        elif self.backend == "opencv":
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass


def setup_hotkeys(state: dict):
    try:
        from pynput import keyboard
    except Exception:
        return None

    def on_press(key):
        if key == keyboard.Key.f8:
            state["paused"] = not state["paused"]
            print("Пауза" if state["paused"] else "Продолжаем")
        elif key in (keyboard.Key.f9, keyboard.Key.esc):
            state["running"] = False
            return False
        return None

    listener = keyboard.Listener(on_press=on_press)
    listener.daemon = True
    listener.start()
    return listener


def main() -> None:
    parser = argparse.ArgumentParser(description="Clicker AI bot")
    parser.add_argument("--model", default="best.pt")
    parser.add_argument("--window", default="Кликер", help="часть заголовка окна игры")
    parser.add_argument("--crop-window", action="store_true", help="захватывать только окно игры")
    parser.add_argument("--preview", action="store_true", help="показывать окно с детекцией")
    parser.add_argument("--dry-run", action="store_true", help="только детекция, без кликов")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.45)
    parser.add_argument("--click-interval", type=float, default=0.08)
    parser.add_argument("--upgrade-interval", type=float, default=3.0)
    parser.add_argument("--bonus-interval", type=float, default=12.0)
    parser.add_argument("--ad-interval", type=float, default=8.0)
    parser.add_argument("--no-game", action="store_true")
    parser.add_argument("--no-upgrades", action="store_true")
    parser.add_argument("--no-bonuses", action="store_true")
    parser.add_argument("--close-ads", action="store_true", help="нажимать кнопку «Отключить рекламу»")
    parser.add_argument("--max-seconds", type=float, default=0.0, help="0 = без ограничения")
    parser.add_argument("--list-windows", action="store_true")
    args = parser.parse_args()

    if args.list_windows:
        list_windows()
        return

    cfg = Config(
        model=args.model,
        window=args.window,
        crop_window=args.crop_window,
        preview=args.preview,
        dry_run=args.dry_run,
        imgsz=args.imgsz,
        conf=args.conf,
        click_interval=args.click_interval,
        upgrade_interval=args.upgrade_interval,
        bonus_interval=args.bonus_interval,
        ad_interval=args.ad_interval,
        click_game=not args.no_game,
        click_upgrades=not args.no_upgrades,
        click_bonuses=not args.no_bonuses,
        close_ads=args.close_ads,
        max_seconds=args.max_seconds,
    )

    model_path = Path(cfg.model)
    if not model_path.exists():
        raise SystemExit(f"Модель не найдена: {model_path}. Сначала обучите её или укажите --model.")

    print(f"Модель: {model_path}")
    model = YOLO(str(model_path))
    print("Классы:", model.names)

    window = find_window(cfg.window)
    if window:
        print(f"Окно игры: {window.title!r} ({window.left},{window.top},{window.width},{window.height})")
    else:
        print(f"Окно «{cfg.window}» не найдено — работаю по всему экрану.")
        print("Подсказка: python app.py --list-windows")

    state = {"running": True, "paused": False}
    listener = setup_hotkeys(state)
    stats = Stats()

    print("Старт. F8 — пауза, F9/Esc — выход." if listener else "Старт. Остановка: Ctrl+C.")
    if cfg.dry_run:
        print("Режим --dry-run: кликов не будет.")
    preview = None
    if cfg.preview:
        try:
            preview = Preview()
        except RuntimeError as exc:
            print(f"Превью недоступно: {exc}")
            cfg.preview = False

    last_log = time.time()
    try:
        while state["running"]:
            if cfg.max_seconds and time.time() - stats.started > cfg.max_seconds:
                break
            if state["paused"]:
                time.sleep(0.1)
                continue

            # область захвата: всё окно игры или весь экран
            region = None
            offset_x = offset_y = 0
            if cfg.crop_window and window:
                region = (window.left, window.top, window.width, window.height)
                offset_x, offset_y = window.left, window.top

            frame = grab_screen(region)
            frame_h, frame_w = frame.shape[:2]
            detections = detect(model, frame, cfg.conf, cfg.imgsz)
            detections.sort(key=lambda d: d[1], reverse=True)
            stats.frames += 1

            # проверяем, что игра в фокусе — иначе не кликаем по чужим окнам
            active_title = ""
            try:
                active = gw.getActiveWindow()
                active_title = (active.title or "") if active else ""
            except Exception:
                pass
            if window is None:
                # окно не нашли — работаем по всему экрану, проверку фокуса пропускаем
                game_active = True
            else:
                game_active = bool(active_title and cfg.window.lower() in active_title.lower())
            can_click = game_active and not cfg.dry_run

            now = time.time()
            best_game = None
            for name, score, box in detections:
                if name == "game" and valid_game_box(box, frame_w, frame_h):
                    if best_game is None or score > best_game[0]:
                        best_game = (score, box)
            if best_game and cfg.click_game and can_click and now - stats.last_game_click >= cfg.click_interval:
                x, y = center(best_game[1])
                pyautogui.click(offset_x + x, offset_y + y)
                stats.clicks += 1
                stats.last_game_click = now

            if cfg.click_upgrades and can_click and now - stats.last_upgrade_click >= cfg.upgrade_interval:
                upgrades = sorted(
                    (d for d in detections if d[0] == "upgrade"),
                    key=lambda d: (d[2][1], d[2][0]),
                )
                if upgrades:
                    box = upgrades[stats.upgrade_idx % len(upgrades)][2]
                    x, y = center(box)
                    pyautogui.click(offset_x + x, offset_y + y)
                    stats.clicks += 1
                    stats.upgrade_idx += 1
                    stats.last_upgrade_click = now

            if cfg.click_bonuses and can_click and now - stats.last_bonus_click >= cfg.bonus_interval:
                bonuses = sorted(
                    (d for d in detections if d[0] == "bonus"),
                    key=lambda d: (d[2][1], d[2][0]),
                )
                if bonuses:
                    box = bonuses[stats.bonus_idx % len(bonuses)][2]
                    x, y = center(box)
                    pyautogui.click(offset_x + x, offset_y + y)
                    stats.clicks += 1
                    stats.bonus_idx += 1
                    stats.last_bonus_click = now

            if cfg.close_ads and can_click and now - stats.last_ad_click >= cfg.ad_interval:
                ads = [d for d in detections if d[0] == "advertising"]
                if ads:
                    x, y = center(ads[0][2])
                    pyautogui.click(offset_x + x, offset_y + y)
                    stats.clicks += 1
                    stats.last_ad_click = now
                    print("Нажал «Отключить рекламу» (страница может открыться).")

            if cfg.preview and preview is not None:
                shown = draw(frame, detections, stats, state["paused"])
                if preview.show(shown) in (ord("q"), 27):
                    break

            if now - last_log >= 1.0:
                names = ", ".join(f"{n}:{sum(1 for d in detections if d[0] == n)}" for n in COLORS)
                print(
                    f"FPS {stats.fps:.1f} | кликов {stats.clicks} | объектов: {names}"
                    + ("" if game_active else " | окно игры не активно")
                )
                last_log = now

    except pyautogui.FailSafeException:
        print("\nСработала защита pyautogui (мышь в углу). Остановлено.")
    except KeyboardInterrupt:
        print("\nОстановлено пользователем.")
    finally:
        if preview is not None:
            preview.close()
        if listener:
            listener.stop()
        print(f"Итог: кадров {stats.frames}, кликов {stats.clicks}, время {time.time() - stats.started:.1f} c.")


if __name__ == "__main__":
    main()
