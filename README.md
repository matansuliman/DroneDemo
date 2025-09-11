# Drone Precision Landing

A MuJoCo + PySide6 desktop app for **precision landing** of a quadrotor on a moving circular pad.  
The drone uses a bottom camera → ArUco detection → simple predictor → PID controller to align and descend.

> **Platform**: Windows • **Python**: 3.11 • **License**: Private (do not distribute)

---

## Quick Start

1. **Create venv (recommended) & install deps**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure** (edit `config.yaml`)
   - Set the MuJoCo scene path (example below uses the repo root):
     ```yaml
     path_to_xml: "skydio_x2/scene.xml"
     ```
   - (Optional) Tweak plotting and camera:
     ```yaml
     plotter:
       ext: png
     camera_streamer:
       resolutions:
         high: [640, 480]
         low:  [320, 240]
       fps: 30
     ```
     
3. **Run**
   ```bash
   python app.py
   ```

---

## What You’ll See

- A GUI with **status readouts** and a **live bottom camera** preview (low‑res).  
- The app runs the **simulation** and **camera streamer** on background threads.  
- When centered and stable above the pad, the controller **descends in phases** and touches down; plots are saved on exit (e.g., `Quadrotor-plot.png`).

---

## Repo Map (key files)

- `app.py` – bootstraps threads: GUI, SimulationRunner and CameraStreamer
- `simulation.py` – main loop, pause/resume/terminate, plotting on exit
- `orchestrators.py` – scene logic (**Follow**), viewer camera
- `models.py` – `Quadrotor`, `Pad` with sensors & logging
- `controllers.py` – PID control & descend phases
- `streamers.py` – offscreen render → detector + GUI
- `detectors.py` / `predictors.py` – ArUco pipeline + simple predictor
- `environment.py` – MuJoCo wrapper, wind/drag helpers
- `Quadrotor.xml`, `Pad.xml`, `scene.xml` – world and assets
- `plots.py`, `logger.py`, `helpers.py`, `noises.py`, `fps.py`

---

## Troubleshooting

- **OpenGL/GLFW errors**: update GPU drivers; ensure a desktop OpenGL context is available.

---

## License

Private (do not distribute)