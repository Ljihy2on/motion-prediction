# Motion Prediction v2

Python + OpenCV real-time object tracking and Kalman motion prediction.

## Setup (macOS)

```bash
cd motion_predictor_v2
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

Your Mac previously detected camera index 1:

```bash
python main.py --mode camera --camera 1
```

**Drag a rectangle around the desired object using the mouse.** Use a visually distinctive object, preferably against a different-looking background. The selected object is tracked with normalized template matching, without relying on color. `r` clears the selection and lets you select again.

```bash
python main.py --mode sim
python main.py --mode sim --headless --frames 180 --metrics results.csv --save demo.mp4
```

## Keyboard

- `q` / Esc: quit
- `r`: reset tracker and history
- `m`: mouse ROI tracking (default for camera)
- `c`: legacy color tracking
- `1`/`2`/`3`: red/blue/green color tracking
- `f`: optical-flow arrows
- `p`: future prediction overlay
- `z`: virtual risk zone
- `[` / `]`: shorten / lengthen prediction horizon

## Notes

- Template matching is a lightweight baseline, **not** object recognition. Strong rotation, scale change, occlusion, and similar backgrounds can cause loss of tracking. When lost, press `r` and reselect.
- The motion estimate is in **image pixels per second**, not physical meters per second.
- The Kalman filter uses a constant-velocity model. Sudden turns can lead to large errors.
- Camera prediction errors use future **observed tracked coordinates**, not independently verified ground truth; they can be misleading if the tracker loses the object.
- Sim mode is retained from v1 and uses a synthetic moving target.
