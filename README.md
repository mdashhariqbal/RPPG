# Webcam Pulse Detector

Non-contact heart-rate (BPM) estimation from a live webcam feed using
remote photoplethysmography (rPPG). Skin's green-channel intensity
oscillates in sync with blood volume changes caused by each heartbeat —
this project detects that signal, cleans it, and turns it into a live BPM
reading.

## How it works

1. **Locate a region of interest (ROI)** each frame — either:
   - **Forehead mode** (default): auto-detected via MediaPipe Face Detection, using the top ~20–30% of the face box.
   - **Hand/Forearm mode**: a fixed on-screen box you place your hand or forearm over.
2. **Record** the mean green-channel intensity of that ROI over time.
3. **Smooth** the signal with a moving average and remove slow drift (e.g. from lighting changes).
4. **Detect peaks** with `scipy.signal.find_peaks`, corresponding to individual heartbeats.
5. **Calculate BPM**: `BPM = 60 / average_peak_to_peak_interval_seconds`.
6. **Display**: live BPM overlay on the camera feed, plus a live waveform plot.

## Requirements

- Python 3.9+
- A webcam

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

```bash
python webcam_pulse.py
```

Optional flags:

```bash
python webcam_pulse.py --camera 1        # use a different camera index
python webcam_pulse.py --mode hand        # start in hand/forearm mode
```

### Controls (camera window must be focused)

| Key | Action |
|-----|--------|
| `q` | Quit — saves the session (CSV + plot) to disk |
| `h` | Toggle between Forehead and Hand/Forearm mode |
| `s` | Save a snapshot of the current data immediately |

### Tips for best results

- Sit in steady, even lighting (avoid strong backlight or flickering light).
- Stay fairly still — motion is the biggest source of noise in this method.
- Give it ~3 seconds of steady data before expecting a stable BPM reading.
- In hand mode, keep your palm/forearm filling the on-screen box.

## Output

On quit (or pressing `s`), the app saves to the working directory:

- `webcam_pulse_<timestamp>.csv` — raw green-channel signal vs. time
- `webcam_pulse_bpm_<timestamp>.csv` — BPM readings over the session
- `webcam_pulse_waveform_<timestamp>.png` — the waveform plot

## Limitations

- Sensitive to motion and lighting changes (inherent to rPPG methods).
- Single-face tracking only (uses the largest/first detected face).
- Not a medical device — for educational/demonstration purposes only.

## References

- [Remote plethysmographic imaging using ambient light (MDPI)](https://www.mdpi.com/2076-3417/10/23/8630)
- [Advancements in Noncontact Video-Based Vital Signs Monitoring (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9735565/pdf/sensors-22-09373.pdf)
- [Measuring Pulse Rate with a Webcam – a Non-contact Method for Evaluating Cardiac Activity](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=7ad15b6fecdb9b2ad49be5bf26efafe22c9a8945)
- [Extracting Blood-Induced Color Changes on the Face for Non-Contact Heart Rate Estimation](https://www.tus.ac.jp/en/mediarelations/archive/20230807_2387.html)
