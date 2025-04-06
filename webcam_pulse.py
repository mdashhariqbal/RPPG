"""
Webcam Pulse Detector
----------------------
Non-contact heart-rate (BPM) estimation from a live webcam feed, based on
remote photoplethysmography (rPPG): skin's green-channel intensity oscillates
in sync with blood volume changes caused by each heartbeat.

Two ROI modes (matches the project write-up):
  1. Forehead  - auto-detected via MediaPipe Face Detection (default)
  2. Hand/Forearm - a fixed on-screen box you place your hand/forearm over

Pipeline:
  1. Locate the ROI each frame (face-based forehead box, or fixed hand box).
  2. Record the mean green-channel intensity of that region over time.
  3. Smooth the signal (moving average) and remove slow drift.
  4. Detect peaks (scipy.signal.find_peaks) corresponding to heartbeats.
  5. Convert average peak-to-peak time into BPM: BPM = 60 / avg_interval_sec.
  6. Show a live waveform plot (matplotlib) alongside the camera feed.

Controls (camera window must be focused):
  q  - quit (saves waveform + BPM history to CSV, and a final plot image)
  h  - toggle between Forehead mode and Hand/Forearm mode
  s  - save a snapshot of the current data immediately (without quitting)

Requirements: opencv-python, mediapipe, numpy, scipy, matplotlib
Install with:
    pip install opencv-python mediapipe numpy scipy matplotlib

Run with:
    python webcam_pulse.py
    python webcam_pulse.py --camera 1        # pick a different camera
    python webcam_pulse.py --mode hand        # start in hand/forearm mode
"""

import argparse
import csv
import datetime
import sys
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
from scipy.signal import find_peaks

import matplotlib.pyplot as plt  # uses whatever interactive backend is available

# ----------------------------------------------------------------------
# Tunable parameters
# ----------------------------------------------------------------------
BUFFER_SECONDS = 10           # how many seconds of signal to keep/analyze
MIN_BPM, MAX_BPM = 45, 180    # physiologically plausible pulse range
SMOOTHING_WINDOW = 5          # moving-average window (frames)
MIN_DETECTION_CONFIDENCE = 0.5
PLOT_UPDATE_EVERY_N_FRAMES = 5  # redraw the live plot every N frames (perf)


def moving_average(values, window):
    """Simple moving average. Output is shorter than input by (window - 1)."""
    if len(values) < window:
        return np.array(values)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def get_forehead_box(detection, frame_w, frame_h):
    """Given a MediaPipe face detection, return (x, y, w, h) of the forehead ROI.

    Isolates the top 20-30% of the detected face box, inset from the sides
    to avoid hairline/background pixels.
    """
    bbox = detection.location_data.relative_bounding_box
    x = int(bbox.xmin * frame_w)
    y = int(bbox.ymin * frame_h)
    w = int(bbox.width * frame_w)
    h = int(bbox.height * frame_h)

    # forehead_h = int(0.3 * h)
    # forehead_y = max(0, y + int(0.08 * h))
    # inset = int(0.20 * w)
    # forehead_x = x + inset
    # forehead_w = max(1, w - 2 * inset)

    offset_y = int(h * 0.20)          # shift up above the face box's top edge (slightly lower than before)
    forehead_y = max(0, y - offset_y)
    forehead_h = int(h * 0.25)        # taller band
    inset = int(0.20 * w)             # slightly wider band, still inset from hairline/temples
    forehead_x = x + inset
    forehead_w = max(1, w - 2 * inset)

    return forehead_x, forehead_y, forehead_w, forehead_h


class PulseApp:
    def __init__(self, camera_index=0, mode="forehead"):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open camera index {camera_index}. "
                "Check that a webcam is connected and not in use by another app."
            )

        self.fps_estimate = self.cap.get(cv2.CAP_PROP_FPS)
        if not self.fps_estimate or self.fps_estimate <= 1 or self.fps_estimate > 120:
            self.fps_estimate = 30  # sane fallback if the camera doesn't report FPS

        buffer_len = int(BUFFER_SECONDS * self.fps_estimate)
        self.times = deque(maxlen=buffer_len)
        self.green_values = deque(maxlen=buffer_len)
        self.bpm_history = []  # (timestamp, bpm) for the whole session, saved on exit

        self.face_detector = mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=MIN_DETECTION_CONFIDENCE
        )

        assert mode in ("forehead", "hand")
        self.mode = mode

        self.start_time = None
        self.current_bpm = None
        self.frame_count = 0

        # ---- Live plot setup ----
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(6, 3))
        (self.line,) = self.ax.plot([], [], color="green", linewidth=1)
        self.ax.set_title("Live Green-Channel Pulse Signal")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Mean green intensity")
        self.ax.grid(True, alpha=0.3)
        self.fig.tight_layout()
        self.fig.show()

    # ------------------------------------------------------------------
    # ROI helpers
    # ------------------------------------------------------------------
    def get_hand_box(self, frame_w, frame_h):
        """Fixed rectangular ROI for hand/forearm mode. User places their
        hand/forearm inside this box (matches Figure 2 in the write-up)."""
        x = int(frame_w * 0.35)
        y = int(frame_h * 0.35)
        w = int(frame_w * 0.30)
        h = int(frame_h * 0.20)
        return x, y, w, h

    # ------------------------------------------------------------------
    # Signal processing
    # ------------------------------------------------------------------
    def compute_bpm(self):
        """Run smoothing + peak detection on the current buffer and return BPM (or None)."""
        if len(self.green_values) < self.fps_estimate * 3:
            return None  # need a few seconds of data first

        raw = np.array(self.green_values)
        smoothed = moving_average(raw, SMOOTHING_WINDOW)
        if len(smoothed) < 2:
            return None

        # Remove slow drift (e.g. lighting changes) by subtracting a longer
        # rolling average, then detect peaks in what's left.
        detrend_window = max(SMOOTHING_WINDOW * 4, 15)
        baseline = moving_average(smoothed, min(detrend_window, len(smoothed)))
        pad = len(smoothed) - len(baseline)
        baseline = np.concatenate([np.full(pad, baseline[0]), baseline]) if pad > 0 else baseline
        detrended = smoothed - baseline

        min_distance = int(self.fps_estimate * 60 / MAX_BPM)
        peaks, _ = find_peaks(detrended, distance=max(1, min_distance))

        if len(peaks) < 2:
            return None

        peak_times = np.array(self.times)[-len(smoothed):][peaks]
        intervals = np.diff(peak_times)
        intervals = intervals[intervals > 0]
        if len(intervals) == 0:
            return None

        avg_interval = np.mean(intervals)
        bpm = 60.0 / avg_interval

        if MIN_BPM <= bpm <= MAX_BPM:
            return bpm
        return None

    # ------------------------------------------------------------------
    # Plotting / saving
    # ------------------------------------------------------------------
    def update_plot(self):
        if len(self.times) < 2:
            return
        t = np.array(self.times)
        g = np.array(self.green_values)
        self.line.set_data(t, g)
        self.ax.set_xlim(t[0], t[-1] + 0.01)
        self.ax.set_ylim(g.min() - 1, g.max() + 1)
        title = "Live Green-Channel Pulse Signal"
        if self.current_bpm:
            title += f"  |  BPM: {self.current_bpm:.1f}"
        self.ax.set_title(title)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def save_session(self):
        """Save the waveform (CSV), BPM history (CSV), and a final plot image."""
        if len(self.times) == 0:
            print("No data collected; nothing to save.")
            return

        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        csv_name = f"webcam_pulse_{stamp}.csv"
        with open(csv_name, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time_sec", "green_channel_mean"])
            for t, g in zip(self.times, self.green_values):
                writer.writerow([t, g])
        print(f"Saved raw signal to {csv_name}")

        if self.bpm_history:
            bpm_csv = f"webcam_pulse_bpm_{stamp}.csv"
            with open(bpm_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["time_sec", "bpm"])
                writer.writerows(self.bpm_history)
            print(f"Saved BPM history to {bpm_csv}")

        plot_name = f"webcam_pulse_waveform_{stamp}.png"
        self.fig.savefig(plot_name, dpi=150, bbox_inches="tight")
        print(f"Saved waveform plot to {plot_name}")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self):
        self.start_time = cv2.getTickCount()
        print("Webcam pulse detector running.")
        print("  q - quit and save session")
        print("  h - toggle Forehead / Hand mode")
        print("  s - save a snapshot immediately")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read frame from camera; exiting.")
                break

            frame = cv2.flip(frame, 1)  # mirror for a natural selfie-view
            h, w, _ = frame.shape
            elapsed = (cv2.getTickCount() - self.start_time) / cv2.getTickFrequency()
            self.frame_count += 1

            roi = None
            roi_color = (0, 255, 0)

            if self.mode == "forehead":
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_detector.process(rgb)
                if results.detections:
                    detection = results.detections[0]  # largest/first face
                    fx, fy, fw, fh = get_forehead_box(detection, w, h)
                    roi = (fx, fy, fx + fw, fy + fh)
                else:
                    cv2.putText(frame, "No face detected", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            else:  # hand mode
                hx, hy, hw, hh = self.get_hand_box(w, h)
                roi = (hx, hy, hx + hw, hy + hh)
                cv2.putText(frame, "Place hand/forearm in box", (hx, hy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if roi is not None:
                x1, y1, x2, y2 = roi
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if x2 > x1 and y2 > y1:
                    region = frame[y1:y2, x1:x2]
                    mean_green = float(np.mean(region[:, :, 1]))  # BGR -> index 1 = Green

                    self.times.append(elapsed)
                    self.green_values.append(mean_green)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), roi_color, 2)

                bpm = self.compute_bpm()
                if bpm is not None:
                    self.current_bpm = bpm
                    self.bpm_history.append((elapsed, bpm))

                label = f"BPM: {self.current_bpm:.1f}" if self.current_bpm else "Calculating BPM..."
                cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (0, 255, 0), 2)

            mode_label = f"Mode: {self.mode.capitalize()}  (press 'h' to switch)"
            cv2.putText(frame, mode_label, (20, h - 50), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 1)
            cv2.putText(frame, "Press 'q' to quit", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 1)
            cv2.imshow("Webcam Pulse Detector", frame)

            if self.frame_count % PLOT_UPDATE_EVERY_N_FRAMES == 0:
                self.update_plot()

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("h"):
                self.mode = "hand" if self.mode == "forehead" else "forehead"
                print(f"Switched to {self.mode} mode")
            elif key == ord("s"):
                self.save_session()

        self.cap.release()
        cv2.destroyAllWindows()
        plt.close(self.fig)
        self.save_session()


def main():
    parser = argparse.ArgumentParser(description="Webcam-based non-contact pulse (BPM) detector.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--mode", choices=["forehead", "hand"], default="forehead",
                         help="Start in forehead (face-detected) or hand/forearm (fixed box) mode")
    args = parser.parse_args()

    try:
        app = PulseApp(camera_index=args.camera, mode=args.mode)
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)

    app.run()


if __name__ == "__main__":
    main()
