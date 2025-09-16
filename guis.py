import numpy as np
from PySide6 import QtWidgets, QtCore
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QWidget

from helpers import *

from environment import ENVIRONMENT
from config import CONFIG
from logger import LOGGER


class GUI(QWidget):
    def __init__(self, simulation):
        super().__init__()
        self._simulation = simulation

        gui_conf = CONFIG["gui"]

        self.setWindowTitle(
            f"{self.__class__.__name__} connected to {simulation.__class__.__name__}"
        )
        self.setGeometry(gui_conf["geometry_top_left_x"],
                         gui_conf["geometry_top_left_y"],
                         gui_conf["geometry_width"],
                         gui_conf["geometry_height"])

        # ============ ROOT ============
        root = QtWidgets.QVBoxLayout(self)

        # ============ LABELS ============
        # camera streamer status
        self._camera_streamer_status = QtWidgets.QLabel("")
        root.addWidget(self._camera_streamer_status)

        # simulation status
        self._simulation_status = QtWidgets.QLabel("")
        root.addWidget(self._simulation_status)

        # camera streamer view
        self._camera_streamer_frame = QtWidgets.QLabel()
        root.addWidget(self._camera_streamer_frame, alignment=QtCore.Qt.AlignCenter)

        # ============ CONTROLS ============
        sliders_row = QtWidgets.QHBoxLayout()

        # pad vx, vy sliders
        vel_box = self._group("Platform Velocity (m/s)")
        gui_pad_conf = gui_conf["pad"]["slider"]

        self._pad_vel_x = self._create_slider(
            gui_pad_conf["low"],
            gui_pad_conf["high"],
            self._simulation.get_pad_vel()[0],
            "Vx",
            on_change=self._set_pad_vel,
            step= gui_pad_conf["step"],
        )
        vel_box.layout().addLayout(self._pad_vel_x["layout"])

        self._pad_vel_y = self._create_slider(
            gui_pad_conf["low"],
            gui_pad_conf["high"],
            self._simulation.get_pad_vel()[0],
            "Vy",
            on_change=self._set_pad_vel,
            step= gui_pad_conf["step"],
        )
        vel_box.layout().addLayout(self._pad_vel_y["layout"])

        sliders_row.addWidget(vel_box, stretch=1)

        # wind (world) vx, vy sliders
        wind_box = self._group("Wind (World, m/s)")
        gui_wind_conf = gui_conf["wind"]["slider"]

        self._wind_x = self._create_slider(
            gui_wind_conf["low"],
            gui_wind_conf["high"],
            ENVIRONMENT.get_wind_vel()[0],
            "Wind Vx",
            on_change=self._set_wind,
            step= gui_wind_conf["step"]
        )
        wind_box.layout().addLayout(self._wind_x["layout"])

        self._wind_y = self._create_slider(
            gui_wind_conf["low"],
            gui_wind_conf["high"],
            ENVIRONMENT.get_wind_vel()[1],
            "Wind Vy",
            on_change= self._set_wind,
            step= gui_wind_conf["step"]
        )
        wind_box.layout().addLayout(self._wind_y["layout"])

        sliders_row.addWidget(wind_box, stretch=1)

        root.addLayout(sliders_row)

        # buttons row: resume/pause, terminate
        buttons_row = QtWidgets.QHBoxLayout()

        self._resume_pause_btn = QtWidgets.QPushButton("Resume")
        self._resume_pause_btn.clicked.connect(self._toggle_resume_pause_resume)
        buttons_row.addWidget(self._resume_pause_btn)

        self._terminate_btn = QtWidgets.QPushButton("Terminate")
        self._terminate_btn.clicked.connect(self._on_terminate)
        buttons_row.addWidget(self._terminate_btn)

        root.addLayout(buttons_row)

        # ============ TICKER ============
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(
            self._sync_resume_pause_label
        )  # Keep resume_pause label synced
        self._timer.start(100)

        LOGGER.info(f"\tGUI: Initiated {self.__class__.__name__}")

    # signals from Simulation and Camera Streamer
    def update_camera_streamer_frame(self, frame):
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_img = QImage(frame.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888)
        self._camera_streamer_frame.setPixmap(QPixmap.fromImage(qt_img))

    def update_simulation_status(self, status: str):
        self._simulation_status.setText(status)

    def update_camera_streamer_status(self, status: str):
        self._camera_streamer_status.setText(status)

    # ---------- Helpers ---------- #
    def _group(self, title: str) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox(title)
        lay = QtWidgets.QVBoxLayout(box)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)
        return box

    def _create_slider(
        self, min_val, max_val, init_val, label, on_change=None, step=0.1
    ):
        layout = QtWidgets.QVBoxLayout()
        caption = QtWidgets.QLabel(f"{label}: {init_val:.2f}")
        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(int(round((max_val - min_val) / step)))
        slider.setValue(int(round((init_val - min_val) / step)))
        slider.setSingleStep(1)

        def _update():
            value = min_val + slider.value() * step
            caption.setText(f"{label}: {value:.2f}")
            if on_change:
                on_change()

        slider.valueChanged.connect(_update)
        layout.addWidget(caption)
        layout.addWidget(slider)
        return {"layout": layout, "slider": slider, "min": min_val, "step": step}

    def _slider_value(self, sdict) -> float:
        return sdict["min"] + sdict["slider"].value() * sdict["step"]

    # controls send to Simulation
    def _set_pad_vel(self):
        new_pad_vel = np.array(
            [
                self._slider_value(self._pad_vel_x),
                self._slider_value(self._pad_vel_y),
                float(self._simulation.get_pad_vel()[2]),  # keep Z unchanged
            ]
        )
        self._simulation.set_pad_vel(new_pad_vel)
        LOGGER.debug(f"GUI: platform velocity = {print_for_gui(new_pad_vel)}")

    def _set_wind(self):
        new_wind_vel = np.array(
            [self._slider_value(self._wind_x), self._slider_value(self._wind_y), 0]
        )
        ENVIRONMENT.enable_wind(True)
        ENVIRONMENT.set_wind(velocity_world= new_wind_vel)
        LOGGER.debug(f"GUI: wind = {print_for_gui(new_wind_vel)}")

    def _toggle_resume_pause_resume(self):
        if self._simulation.is_running():
            LOGGER.debug("GUI: pressed pause")
            self._simulation.pause()
        else:
            LOGGER.debug("GUI: pressed resume")
            self._simulation.resume()

    def _sync_resume_pause_label(self):
        running = self._simulation.is_running()
        want = "Pause" if running else "Resume"
        if self._resume_pause_btn.text() != want:
            self._resume_pause_btn.setText(want)

    def _on_terminate(self):
        LOGGER.debug("GUI: pressed terminate")
        self._simulation.terminate()
        QtCore.QTimer.singleShot(200, QApplication.quit)
