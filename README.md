# Drone Precision Landing

A MuJoCo + PySide6 desktop app for **precision landing** of a quadrotor on a moving circular pad.  
The drone uses a bottom camera → ArUco detection → simple predictor → PID controller to align and descend.

> **Platform**: Windows • **Python**: 3.11

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
       frequency: 30
     ```
     
3. **Run**
   ```bash
   python app.py
   ```

---

## What You’ll See

- A GUI with **status readouts** and a **live preview** (low‑res).  
- The app runs the **simulation** and **camera streamer** on background threads.  
- When centered and stable above the pad, the controller **descends in phases** and touches down; plots are saved on exit (e.g., `Quadrotor-plot.png`).

---

## Repo Map (key files)

- `app.py` – bootstraps threads: GUI, SimulationRunner and CameraStreamer
- `simulation.py` – main loop, pause/resume/terminate
- `gui.py` - graphical user interface
- `streamers.py` – offscreen render to detector(through simulation) and GUI
- `fps.py` - maintaining frequency of loops
- `orchestrators.py` – scene logic (**Follow orchestrator**)
- `models.py` – `Quadrotor`, `Pad` physical object with sensors & logging
- `controllers.py` – non-physical, controlling the physical objects
  (**QuadrotorController** - PID control & descend phases)
- `sensors.py` - basic MuJoCo sensor and advanced sensors
- `noises.py` - synthetic noise for sensors
- `predictors.py` - simple predictor based on history
- `detectors.py` – detects pad from streamer (ArUco marker detector)
- `environment.py` – MuJoCo wrapper
- `plots.py` - plotting logs on png images, called on exit
- `logger.py` - global log
- `config.py` / `config.yaml` - global config from yaml
- `helpers.py` - global helper functions
- `/skydio_x2` - MuJoCo xml files (world and objects), assets

---

## More Info

- **Presentation.pdf**
- **ClassDiagram.pdf**

---

## Troubleshooting

- **OpenGL/GLFW errors**: update GPU drivers; ensure a desktop OpenGL context is available.
