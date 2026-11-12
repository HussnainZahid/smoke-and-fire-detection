"""
Smoke and Fire Detection
Simple Desktop Dashboard Application (Run directly with: python app.py)

Automatically loads the video placed inside `data/input/`, detects both Fire and Smoke
frame-by-frame using a Hybrid Deep Learning (YOLO26n `best.pt`) + Multi-Scale /
Chromatic Flame Verification pipeline so Fire inside dense Smoke plumes is never missed,
draws Red (FIRE) and Cyan (SMOKE) bounding boxes, triggers visual & sound alerts,
and saves the annotated output video to `data/output/`.
"""

from pathlib import Path
import shutil
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO

try:
    import winsound
except ImportError:
    winsound = None


# ---------------------------------------------------------------------------
# Simple Project Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "data" / "input"
OUTPUT_DIR = BASE_DIR / "data" / "output"
MODELS_DIR = BASE_DIR / "models"

INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}

# Bounding Box Colors in BGR (OpenCV)
COLOR_FIRE_BGR = (25, 25, 245)    # Bright Red for FIRE
COLOR_SMOKE_BGR = (240, 195, 0)   # Cyan/Blue for SMOKE


def find_model_path() -> Optional[Path]:
    """Finds best.pt in models/best.pt or the project root."""
    candidates = [
        MODELS_DIR / "best.pt",
        BASE_DIR / "best.pt",
    ]
    for path in candidates:
        if path.exists() and path.is_file():
            return path
    return None


def find_input_videos() -> List[Path]:
    """Returns all video files placed inside data/input/."""
    if not INPUT_DIR.exists():
        return []
    return [
        p for p in sorted(INPUT_DIR.iterdir())
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    ]


def compute_iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    """Computes Intersection-over-Union (IoU) between two (x1, y1, x2, y2) boxes."""
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    inter_w = max(0, x_b - x_a)
    inter_h = max(0, y_b - y_a)
    inter_area = float(inter_w * inter_h)
    if inter_area <= 0.0:
        return 0.0

    area_a = float(max(1, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])))
    area_b = float(max(1, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])))
    return inter_area / max(1.0, area_a + area_b - inter_area)


def detect_chromatic_flame_regions(
    frame_bgr: np.ndarray,
    smoke_boxes: List[Tuple[float, Tuple[int, int, int, int]]],
    existing_fire_boxes: List[Tuple[float, Tuple[int, int, int, int]]],
    fire_thresh: float,
) -> List[Tuple[float, Tuple[int, int, int, int]]]:
    """
    Complements YOLO26n by detecting active glowing flames (using RGB + HSV + YCrCb
    flame-core segmentation) that are partially occluded by dense black smoke.
    """
    h, w = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    b, g, r = cv2.split(frame_bgr)

    r_i = r.astype(np.int16)
    g_i = g.astype(np.int16)
    b_i = b.astype(np.int16)

    # Strict multi-space flame signature:
    # 1. Red dominance & warm gradient (R > G > B)
    mask_rgb = (r > 180) & ((r_i - g_i) >= 22) & ((g_i - b_i) >= 12)
    # 2. High saturation and high brightness in Red-Orange-Yellow Hue [0..30]
    mask_hsv = (hsv[:, :, 0] <= 30) & (hsv[:, :, 1] >= 115) & (hsv[:, :, 2] >= 175)
    # 3. High red-difference chroma (Cr) and low blue-difference chroma (Cb)
    mask_ycrcb = (ycrcb[:, :, 1] >= 148) & (ycrcb[:, :, 2] <= 122)
    # 4. Hot inner core verification (intense bright orange/yellow/white-hot pixels)
    hot_core_mask = (r > 225) & (g > 110) & (hsv[:, :, 2] >= 215)

    combined = ((mask_rgb & mask_hsv & mask_ycrcb).astype(np.uint8)) * 255
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, close_k, iterations=2)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, open_k, iterations=1)

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_flame_area = max(140.0, 0.0015 * float(w * h))

    extra_fires: List[Tuple[float, Tuple[int, int, int, int]]] = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_flame_area:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w - 1, x + bw)
        y2 = min(h - 1, y + bh)
        if x2 <= x1 or y2 <= y1:
            continue

        cand_box = (x1, y1, x2, y2)

        # Skip if already covered by a YOLO fire box
        if any(compute_iou(cand_box, fbox) > 0.25 for _, fbox in existing_fire_boxes):
            continue

        # Check inner hot-core ratio inside the bounding box
        roi_hot = hot_core_mask[y1:y2, x1:x2]
        roi_flame = combined[y1:y2, x1:x2] > 0
        flame_pixels = max(1, int(np.count_nonzero(roi_flame)))
        hot_ratio = float(np.count_nonzero(roi_hot & roi_flame)) / float(flame_pixels)

        # Check if this flame region touches or lies inside a detected smoke plume
        near_smoke = False
        for _, (sx1, sy1, sx2, sy2) in smoke_boxes:
            expanded_smoke = (
                max(0, sx1 - 30),
                max(0, sy1 - 20),
                min(w - 1, sx2 + 30),
                min(h - 1, sy2 + int(0.25 * (sy2 - sy1))),
            )
            if compute_iou(cand_box, expanded_smoke) > 0.01:
                near_smoke = True
                break

        # Require either proximity to a smoke plume OR a very strong hot flame core
        if not near_smoke and hot_ratio < 0.22:
            continue

        # Compute realistic confidence score from flame intensity, hot core ratio, and smoke context
        mean_v = float(np.mean(hsv[y1:y2, x1:x2, 2][roi_flame])) / 255.0
        mean_s = float(np.mean(hsv[y1:y2, x1:x2, 1][roi_flame])) / 255.0
        context_bonus = 0.12 if near_smoke else 0.0
        est_conf = min(0.96, max(0.55, 0.48 + 0.25 * mean_v + 0.15 * mean_s + 0.12 * hot_ratio + context_bonus))

        if est_conf >= fire_thresh:
            extra_fires.append((round(est_conf, 4), cand_box))

    return extra_fires


class SmokeAndFireDashboard:
    """
    Clean, single-window Desktop Dashboard for Smoke and Fire Detection.
    Opens immediately when you run `python app.py`.
    """

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Smoke and Fire Detection - Live Dashboard")
        self.root.geometry("1220x760")
        self.root.minsize(1024, 680)
        self.root.configure(bg="#141824")

        # State variables
        self.model: Optional[YOLO] = None
        self.model_path: Optional[Path] = find_model_path()
        self.class_names: Dict[int, str] = {0: "fire", 1: "smoke"}

        self.input_videos: List[Path] = find_input_videos()
        self.current_video_path: Optional[Path] = self.input_videos[0] if self.input_videos else None

        self.is_running: bool = False
        self.is_paused: bool = False
        self.stop_requested: bool = False

        # Thresholds (default 35% for Fire and 45% for Smoke)
        self.fire_conf_var = tk.IntVar(value=35)
        self.smoke_conf_var = tk.IntVar(value=45)
        self.sound_enabled_var = tk.BooleanVar(value=True)

        # Counters
        self.total_frames_processed = 0
        self.total_fire_detections = 0
        self.total_smoke_detections = 0
        self.last_beep_time = 0.0

        self._build_ui()
        self._load_yolo_model()
        self._refresh_video_list()

        # Automatically start playing the video in data/input/ if both model and video exist
        if self.model is not None and self.current_video_path is not None:
            self.root.after(400, self.start_detection)

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # -----------------------------------------------------------------------
    # UI Construction
    # -----------------------------------------------------------------------
    def _build_ui(self) -> None:
        header = tk.Frame(self.root, bg="#1E2538", height=64, padx=20, pady=10)
        header.pack(fill=tk.X)

        title_lbl = tk.Label(
            header,
            text="🔥 Smoke and Fire Detection Dashboard",
            font=("Segoe UI", 18, "bold"),
            fg="#FFFFFF",
            bg="#1E2538",
        )
        title_lbl.pack(side=tk.LEFT)

        self.model_status_lbl = tk.Label(
            header,
            text="Model: Checking...",
            font=("Segoe UI", 10, "bold"),
            fg="#FFD166",
            bg="#1E2538",
            padx=12,
        )
        self.model_status_lbl.pack(side=tk.RIGHT)

        self.alert_banner = tk.Label(
            self.root,
            text="SYSTEM READY — Place your video in 'data/input/' and click Start Detection",
            font=("Segoe UI", 12, "bold"),
            fg="#E2E8F0",
            bg="#2B354F",
            pady=8,
        )
        self.alert_banner.pack(fill=tk.X)

        main_frame = tk.Frame(self.root, bg="#141824", padx=14, pady=12)
        main_frame.pack(fill=tk.BOTH, expand=True)

        video_container = tk.Frame(main_frame, bg="#0B0E17", bd=2, relief=tk.RIDGE)
        video_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 12))

        self.video_label = tk.Label(
            video_container,
            text="No Video Playing\n\nPut an MP4/AVI video inside:\ndata/input/",
            font=("Segoe UI", 14),
            fg="#94A3B8",
            bg="#0B0E17",
            justify=tk.CENTER,
        )
        self.video_label.pack(fill=tk.BOTH, expand=True)

        side_panel = tk.Frame(main_frame, bg="#1E2538", width=340, padx=16, pady=14)
        side_panel.pack(side=tk.RIGHT, fill=tk.Y)
        side_panel.pack_propagate(False)

        tk.Label(
            side_panel,
            text="1. Video in data/input/",
            font=("Segoe UI", 11, "bold"),
            fg="#38BDF8",
            bg="#1E2538",
        ).pack(anchor="w", pady=(0, 6))

        self.video_combo = ttk.Combobox(side_panel, state="readonly", font=("Segoe UI", 10))
        self.video_combo.pack(fill=tk.X, pady=(0, 6))
        self.video_combo.bind("<<ComboboxSelected>>", self._on_video_selected)

        btn_row_input = tk.Frame(side_panel, bg="#1E2538")
        btn_row_input.pack(fill=tk.X, pady=(0, 12))

        tk.Button(
            btn_row_input,
            text="🔄 Refresh data/input",
            command=self._refresh_video_list,
            bg="#334155",
            fg="#FFFFFF",
            font=("Segoe UI", 9, "bold"),
            relief=tk.FLAT,
            padx=8,
            pady=5,
        ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 4))

        tk.Button(
            btn_row_input,
            text="📂 Browse Video",
            command=self._browse_video_file,
            bg="#334155",
            fg="#FFFFFF",
            font=("Segoe UI", 9, "bold"),
            relief=tk.FLAT,
            padx=8,
            pady=5,
        ).pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=(4, 0))

        tk.Label(
            side_panel,
            text="2. Detection Controls",
            font=("Segoe UI", 11, "bold"),
            fg="#38BDF8",
            bg="#1E2538",
        ).pack(anchor="w", pady=(4, 6))

        self.start_btn = tk.Button(
            side_panel,
            text="▶ Start Detection",
            command=self.start_detection,
            bg="#16A34A",
            fg="#FFFFFF",
            font=("Segoe UI", 11, "bold"),
            relief=tk.FLAT,
            pady=7,
        )
        self.start_btn.pack(fill=tk.X, pady=(0, 6))

        ctrl_row = tk.Frame(side_panel, bg="#1E2538")
        ctrl_row.pack(fill=tk.X, pady=(0, 12))

        self.pause_btn = tk.Button(
            ctrl_row,
            text="⏸ Pause / Resume",
            command=self.toggle_pause,
            bg="#D97706",
            fg="#FFFFFF",
            font=("Segoe UI", 9, "bold"),
            relief=tk.FLAT,
            pady=6,
        )
        self.pause_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 4))

        self.stop_btn = tk.Button(
            ctrl_row,
            text="⏹ Stop",
            command=self.stop_detection,
            bg="#DC2626",
            fg="#FFFFFF",
            font=("Segoe UI", 9, "bold"),
            relief=tk.FLAT,
            pady=6,
        )
        self.stop_btn.pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=(4, 0))

        tk.Label(
            side_panel,
            text="3. Confidence Thresholds",
            font=("Segoe UI", 11, "bold"),
            fg="#38BDF8",
            bg="#1E2538",
        ).pack(anchor="w", pady=(4, 4))

        self.fire_slider_lbl = tk.Label(
            side_panel,
            text=f"🔥 Fire Threshold: {self.fire_conf_var.get()}%",
            font=("Segoe UI", 9, "bold"),
            fg="#F87171",
            bg="#1E2538",
        )
        self.fire_slider_lbl.pack(anchor="w")

        fire_scale = tk.Scale(
            side_panel,
            from_=15,
            to=95,
            orient=tk.HORIZONTAL,
            variable=self.fire_conf_var,
            command=lambda v: self.fire_slider_lbl.config(text=f"🔥 Fire Threshold: {v}%"),
            bg="#1E2538",
            fg="#FFFFFF",
            highlightthickness=0,
            troughcolor="#0F172A",
            showvalue=False,
        )
        fire_scale.pack(fill=tk.X, pady=(0, 6))

        self.smoke_slider_lbl = tk.Label(
            side_panel,
            text=f"💨 Smoke Threshold: {self.smoke_conf_var.get()}%",
            font=("Segoe UI", 9, "bold"),
            fg="#38BDF8",
            bg="#1E2538",
        )
        self.smoke_slider_lbl.pack(anchor="w")

        smoke_scale = tk.Scale(
            side_panel,
            from_=15,
            to=95,
            orient=tk.HORIZONTAL,
            variable=self.smoke_conf_var,
            command=lambda v: self.smoke_slider_lbl.config(text=f"💨 Smoke Threshold: {v}%"),
            bg="#1E2538",
            fg="#FFFFFF",
            highlightthickness=0,
            troughcolor="#0F172A",
            showvalue=False,
        )
        smoke_scale.pack(fill=tk.X, pady=(0, 8))

        tk.Checkbutton(
            side_panel,
            text="🔊 Enable Audible Beep Alert",
            variable=self.sound_enabled_var,
            bg="#1E2538",
            fg="#E2E8F0",
            selectcolor="#0F172A",
            activebackground="#1E2538",
            activeforeground="#FFFFFF",
            font=("Segoe UI", 9, "bold"),
        ).pack(anchor="w", pady=(0, 12))

        tk.Label(
            side_panel,
            text="4. Live Detection Statistics",
            font=("Segoe UI", 11, "bold"),
            fg="#38BDF8",
            bg="#1E2538",
        ).pack(anchor="w", pady=(4, 6))

        stats_box = tk.Frame(side_panel, bg="#0F172A", padx=12, pady=10)
        stats_box.pack(fill=tk.X)

        self.stat_frame_lbl = tk.Label(
            stats_box, text="Frames Processed: 0 / 0", font=("Segoe UI", 10), fg="#E2E8F0", bg="#0F172A"
        )
        self.stat_frame_lbl.pack(anchor="w", pady=2)

        self.stat_fps_lbl = tk.Label(
            stats_box, text="Processing Speed: 0.0 FPS", font=("Segoe UI", 10), fg="#E2E8F0", bg="#0F172A"
        )
        self.stat_fps_lbl.pack(anchor="w", pady=2)

        self.stat_fire_lbl = tk.Label(
            stats_box, text="🔥 Fire Detections: 0", font=("Segoe UI", 10, "bold"), fg="#F87171", bg="#0F172A"
        )
        self.stat_fire_lbl.pack(anchor="w", pady=2)

        self.stat_smoke_lbl = tk.Label(
            stats_box, text="💨 Smoke Detections: 0", font=("Segoe UI", 10, "bold"), fg="#38BDF8", bg="#0F172A"
        )
        self.stat_smoke_lbl.pack(anchor="w", pady=2)

        self.output_info_lbl = tk.Label(
            side_panel,
            text="Annotated output saved in:\ndata/output/",
            font=("Segoe UI", 8),
            fg="#94A3B8",
            bg="#1E2538",
            justify=tk.LEFT,
            wraplength=300,
        )
        self.output_info_lbl.pack(anchor="w", pady=(12, 0))

    # -----------------------------------------------------------------------
    # Model & Input Video Loading
    # -----------------------------------------------------------------------
    def _load_yolo_model(self) -> None:
        self.model_path = find_model_path()
        if self.model_path is None:
            self.model_status_lbl.config(
                text="Model Missing! Put best.pt in models/best.pt", fg="#F87171"
            )
            return

        try:
            self.model = YOLO(str(self.model_path))
            raw_names = getattr(self.model, "names", {0: "fire", 1: "smoke"})
            self.class_names = {int(k): str(v).strip().lower() for k, v in raw_names.items()}
            self.model_status_lbl.config(
                text=f"✅ Model Loaded: {self.model_path.name}", fg="#4ADE80"
            )
        except Exception as exc:
            self.model = None
            self.model_status_lbl.config(text=f"Model Error: {exc}", fg="#F87171")

    def _refresh_video_list(self) -> None:
        self.input_videos = find_input_videos()
        names = [p.name for p in self.input_videos]
        self.video_combo["values"] = names
        if self.input_videos:
            self.video_combo.current(0)
            self.current_video_path = self.input_videos[0]
            self.alert_banner.config(
                text=f"Ready to analyze: data/input/{self.current_video_path.name}",
                bg="#2B354F",
                fg="#E2E8F0",
            )
        else:
            self.video_combo.set("No video found in data/input/")
            self.current_video_path = None

    def _on_video_selected(self, _event: object = None) -> None:
        idx = self.video_combo.current()
        if 0 <= idx < len(self.input_videos):
            self.current_video_path = self.input_videos[idx]

    def _browse_video_file(self) -> None:
        chosen = filedialog.askopenfilename(
            title="Select Video File",
            initialdir=str(INPUT_DIR),
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv *.wmv"), ("All Files", "*.*")],
        )
        if chosen:
            chosen_path = Path(chosen)
            dest_path = INPUT_DIR / chosen_path.name
            if chosen_path.resolve() != dest_path.resolve():
                shutil.copy2(chosen_path, dest_path)
            self._refresh_video_list()
            for i, p in enumerate(self.input_videos):
                if p.name == dest_path.name:
                    self.video_combo.current(i)
                    self.current_video_path = p
                    break

    # -----------------------------------------------------------------------
    # Detection Loop
    # -----------------------------------------------------------------------
    def start_detection(self) -> None:
        if self.is_running:
            self.is_paused = False
            return

        if self.model is None:
            self._load_yolo_model()
            if self.model is None:
                messagebox.showerror(
                    "Model Not Found",
                    f"Please place your trained 'best.pt' file inside:\n{MODELS_DIR / 'best.pt'}\nor\n{BASE_DIR / 'best.pt'}",
                )
                return

        if self.current_video_path is None or not self.current_video_path.exists():
            self._refresh_video_list()
            if self.current_video_path is None:
                messagebox.showwarning(
                    "No Input Video Found",
                    f"Please put a video file (.mp4, .avi, .mov, .mkv) inside:\n{INPUT_DIR}",
                )
                return

        self.is_running = True
        self.is_paused = False
        self.stop_requested = False
        self.total_frames_processed = 0
        self.total_fire_detections = 0
        self.total_smoke_detections = 0

        self.start_btn.config(state=tk.DISABLED)
        threading.Thread(target=self._process_video_worker, daemon=True).start()

    def toggle_pause(self) -> None:
        if self.is_running:
            self.is_paused = not self.is_paused

    def stop_detection(self) -> None:
        if self.is_running:
            self.stop_requested = True
            self.is_paused = False

    def _play_alarm_beep(self) -> None:
        if not self.sound_enabled_var.get():
            return
        now = time.monotonic()
        if now - self.last_beep_time >= 1.5:
            self.last_beep_time = now
            if winsound is not None:
                threading.Thread(target=lambda: winsound.Beep(1100, 280), daemon=True).start()

    def _process_video_worker(self) -> None:
        video_path = self.current_video_path
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            self.is_running = False
            self.root.after(0, lambda: self.start_btn.config(state=tk.NORMAL))
            self.root.after(
                0,
                lambda: messagebox.showerror("Video Error", f"Could not open video:\n{video_path}"),
            )
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0 or fps > 120 or np.isnan(fps):
            fps = 25.0
        total_frames = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

        out_path = OUTPUT_DIR / f"annotated_{video_path.stem}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

        self.root.after(
            0,
            lambda: self.output_info_lbl.config(
                text=f"Saving output video to:\ndata/output/{out_path.name}"
            ),
        )

        start_wall = time.perf_counter()

        try:
            while not self.stop_requested:
                if self.is_paused:
                    time.sleep(0.05)
                    continue

                ret, frame_bgr = cap.read()
                if not ret or frame_bgr is None:
                    break

                self.total_frames_processed += 1
                fire_thresh = self.fire_conf_var.get() / 100.0
                smoke_thresh = self.smoke_conf_var.get() / 100.0

                # 1. Run YOLO26n inference down to a low candidate floor (0.12)
                results = self.model.predict(
                    source=frame_bgr,
                    conf=0.12,
                    iou=0.45,
                    agnostic_nms=False,
                    verbose=False,
                )

                frame_fire_list: List[Tuple[float, Tuple[int, int, int, int]]] = []
                frame_smoke_list: List[Tuple[float, Tuple[int, int, int, int]]] = []

                if results and len(results) > 0:
                    boxes = results[0].boxes
                    if boxes is not None and len(boxes) > 0:
                        xyxy_arr = boxes.xyxy.cpu().numpy()
                        conf_arr = boxes.conf.cpu().numpy()
                        cls_arr = boxes.cls.cpu().numpy()

                        for xyxy, conf_val, cls_val in zip(xyxy_arr, conf_arr, cls_arr):
                            conf_f = float(conf_val)
                            cls_id = int(cls_val)
                            cls_name = self.class_names.get(cls_id, "").lower()
                            x1, y1, x2, y2 = [int(round(v)) for v in xyxy[:4]]

                            if cls_name == "fire" and conf_f >= fire_thresh:
                                frame_fire_list.append((conf_f, (x1, y1, x2, y2)))
                            elif cls_name == "smoke" and conf_f >= smoke_thresh:
                                frame_smoke_list.append((conf_f, (x1, y1, x2, y2)))

                # 2. Hybrid Chromatic Flame Verification:
                # Detects active glowing flames inside/under dense smoke plumes that YOLO26n
                # misses when fire and smoke overlap in the same region.
                extra_fires = detect_chromatic_flame_regions(
                    frame_bgr=frame_bgr,
                    smoke_boxes=frame_smoke_list,
                    existing_fire_boxes=frame_fire_list,
                    fire_thresh=fire_thresh,
                )
                if extra_fires:
                    frame_fire_list.extend(extra_fires)

                self.total_fire_detections += len(frame_fire_list)
                self.total_smoke_detections += len(frame_smoke_list)

                annotated = self._draw_boxes(frame_bgr, frame_fire_list, frame_smoke_list)
                if writer.isOpened():
                    writer.write(annotated)

                elapsed = max(0.001, time.perf_counter() - start_wall)
                proc_fps = self.total_frames_processed / elapsed

                if frame_fire_list or frame_smoke_list:
                    self._play_alarm_beep()

                self._update_gui_frame(
                    annotated,
                    self.total_frames_processed,
                    total_frames,
                    proc_fps,
                    frame_fire_list,
                    frame_smoke_list,
                )

        finally:
            cap.release()
            writer.release()
            self.is_running = False
            self.root.after(0, lambda: self.start_btn.config(state=tk.NORMAL))
            self.root.after(
                0,
                lambda: self.alert_banner.config(
                    text=f"✅ Finished! Output video saved to: data/output/{out_path.name}",
                    bg="#15803D",
                    fg="#FFFFFF",
                ),
            )

    @staticmethod
    def _draw_boxes(
        frame_bgr: np.ndarray,
        fire_dets: List[Tuple[float, Tuple[int, int, int, int]]],
        smoke_dets: List[Tuple[float, Tuple[int, int, int, int]]],
    ) -> np.ndarray:
        canvas = frame_bgr.copy()
        h, w = canvas.shape[:2]
        scale = max(0.55, min(2.0, min(w, h) / 640.0))
        thickness = max(2, int(round(2.5 * scale)))
        font_scale = max(0.55, 0.65 * scale)
        font_thick = max(1, int(round(2.0 * scale)))
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Draw SMOKE boxes first (Cyan/Blue)
        for conf, (x1, y1, x2, y2) in smoke_dets:
            cv2.rectangle(canvas, (x1, y1), (x2, y2), COLOR_SMOKE_BGR, thickness)
            label = f"SMOKE {conf * 100:.1f}%"
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, font_thick)
            top_y = max(th + 8, y1)
            cv2.rectangle(canvas, (x1, top_y - th - 8), (x1 + tw + 10, top_y), COLOR_SMOKE_BGR, -1)
            cv2.putText(canvas, label, (x1 + 5, top_y - 5), font, font_scale, (10, 10, 10), font_thick, cv2.LINE_AA)

        # Draw FIRE boxes on top (Bright Red) so Fire inside a Smoke box is always clearly visible
        for conf, (x1, y1, x2, y2) in fire_dets:
            cv2.rectangle(canvas, (x1, y1), (x2, y2), COLOR_FIRE_BGR, thickness + 1)
            label = f"FIRE {conf * 100:.1f}%"
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, font_thick)
            top_y = max(th + 8, y1)
            cv2.rectangle(canvas, (x1, top_y - th - 8), (x1 + tw + 10, top_y), COLOR_FIRE_BGR, -1)
            cv2.putText(canvas, label, (x1 + 5, top_y - 5), font, font_scale, (255, 255, 255), font_thick, cv2.LINE_AA)

        return canvas

    def _update_gui_frame(
        self,
        annotated_bgr: np.ndarray,
        frame_idx: int,
        total_frames: int,
        fps: float,
        fire_dets: List[Tuple[float, Tuple[int, int, int, int]]],
        smoke_dets: List[Tuple[float, Tuple[int, int, int, int]]],
    ) -> None:
        lbl_w = max(480, self.video_label.winfo_width() - 8)
        lbl_h = max(320, self.video_label.winfo_height() - 8)
        fh, fw = annotated_bgr.shape[:2]
        ratio = min(lbl_w / float(fw), lbl_h / float(fh))
        new_w = max(1, int(fw * ratio))
        new_h = max(1, int(fh * ratio))

        resized_bgr = cv2.resize(annotated_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        def _render() -> None:
            tk_img = ImageTk.PhotoImage(image=pil_img)
            self.video_label.configure(image=tk_img, text="")
            self.video_label.image = tk_img

            self.stat_frame_lbl.config(text=f"Frames Processed: {frame_idx} / {total_frames}")
            self.stat_fps_lbl.config(text=f"Processing Speed: {fps:.1f} FPS")
            self.stat_fire_lbl.config(text=f"🔥 Fire Detections: {self.total_fire_detections}")
            self.stat_smoke_lbl.config(text=f"💨 Smoke Detections: {self.total_smoke_detections}")

            if fire_dets and smoke_dets:
                max_f = max(c for c, _ in fire_dets) * 100
                max_s = max(c for c, _ in smoke_dets) * 100
                self.alert_banner.config(
                    text=f"🚨 CRITICAL ALERT: FIRE ({max_f:.1f}%) & SMOKE ({max_s:.1f}%) DETECTED!",
                    bg="#DC2626",
                    fg="#FFFFFF",
                )
            elif fire_dets:
                max_f = max(c for c, _ in fire_dets) * 100
                self.alert_banner.config(
                    text=f"🚨 FIRE ALERT: FIRE DETECTED ({max_f:.1f}% Confidence)!",
                    bg="#DC2626",
                    fg="#FFFFFF",
                )
            elif smoke_dets:
                max_s = max(c for c, _ in smoke_dets) * 100
                self.alert_banner.config(
                    text=f"⚠️ SMOKE ALERT: SMOKE DETECTED ({max_s:.1f}% Confidence)!",
                    bg="#0284C7",
                    fg="#FFFFFF",
                )
            else:
                self.alert_banner.config(
                    text="🟢 Monitoring Video — No Fire or Smoke Detected",
                    bg="#2B354F",
                    fg="#E2E8F0",
                )

        self.root.after(0, _render)

    def _on_close(self) -> None:
        self.stop_requested = True
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    app = SmokeAndFireDashboard(root)
    root.mainloop()


if __name__ == "__main__":
    main()