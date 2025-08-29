# guis.py — minimal control panel
import sys
import numpy as np
from PySide6 import QtWidgets, QtCore
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QWidget

class GUI(QWidget):
    def __init__(self, simulation, camera_streamer):
        super().__init__()
        self.simulation = simulation
        self.env = simulation.orchestrator.env
        self.objects = simulation.orchestrator.objects

        self.camera_streamer = camera_streamer
        self.landing_active = False
        self.is_paused = False

        self.setWindowTitle(f'{self.__class__.__name__}')
        self.setGeometry(0, 0, 500, 500)

        # ============ ROOT LAYOUT ============
        root = QtWidgets.QVBoxLayout(self)

        self.marker_detection_label = QtWidgets.QLabel("markers detection:")
        root.addWidget(self.marker_detection_label)

        # ============ CAMERA ============
        self.camera_label = QtWidgets.QLabel()
        self.camera_label.setStyleSheet("background:#000;")
        root.addWidget(self.camera_label, alignment=QtCore.Qt.AlignCenter)

        # ============ CONTROLS STRIP ============
        # Velocity sliders (platform Vx, Vy)
        vel_box = self._group("Platform Velocity (m/s)")
        self.vel_x = self._create_slider(-5.0, 5.0, self.objects['platform_controller'].velocity[0],
                                         "Vx", on_change=self._apply_velocity, step=0.1)
        self.vel_y = self._create_slider(-5.0, 5.0, self.objects['platform_controller'].velocity[1],
                                         "Vy", on_change=self._apply_velocity, step=0.1)
        vel_box.layout().addLayout(self.vel_x['layout'])
        vel_box.layout().addLayout(self.vel_y['layout'])

        # Wind sliders (world Vx, Vy) — sliders as requested
        wind_box = self._group("Wind (world, m/s)")
        self.wind_x = self._create_slider(-10.0, 10.0, 0.0, "Wind Vx", on_change=self._apply_wind, step=0.1)
        self.wind_y = self._create_slider(-10.0, 10.0, 0.0, "Wind Vy", on_change=self._apply_wind, step=0.1)
        wind_box.layout().addLayout(self.wind_x['layout'])
        wind_box.layout().addLayout(self.wind_y['layout'])

        # Buttons row: Pause/Resume, Terminate, Land
        buttons = QtWidgets.QHBoxLayout()
        self.pause_btn = QtWidgets.QPushButton("Resume")
        self.pause_btn.clicked.connect(self.toggle_pause_resume)
        buttons.addWidget(self.pause_btn)

        self.terminate_btn = QtWidgets.QPushButton("Terminate")
        self.terminate_btn.clicked.connect(self._on_terminate)
        buttons.addWidget(self.terminate_btn)

        self.land_btn = QtWidgets.QPushButton("Land")
        self.land_btn.setStyleSheet("background:#d00; color:white; font-weight:bold;")
        self.land_btn.clicked.connect(self._on_land)
        buttons.addWidget(self.land_btn)

        # Pack groups + buttons
        groups_row = QtWidgets.QHBoxLayout()
        groups_row.addWidget(vel_box, stretch=1)
        groups_row.addWidget(wind_box, stretch=1)
        root.addLayout(groups_row)
        root.addLayout(buttons)

        # ============ TICKER ============
        # Keep pause label synced; if landing is active, re-apply landing target
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._sync_pause_label)
        self.timer.timeout.connect(self._sync_land_btn)
        self.timer.start(100)

    # ---------- Camera ----------
    def update_camera_view(self, frame):
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_img = QImage(frame.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888)
        self.camera_label.setPixmap(QPixmap.fromImage(qt_img))

    # Keep for compatibility with app.py’s connection, but no UI text (by request).
    def update_marker_detection(self, result: str):
        if result:
            self.marker_detection_label.setText(f"Marker Detection: {result}")

    # ---------- Helpers ----------
    def _group(self, title: str) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox(title)
        lay = QtWidgets.QVBoxLayout(box)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)
        return box

    def _create_slider(self, min_val, max_val, init_val, label, on_change=None, step=0.1):
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
        return {'layout': layout, 'slider': slider, 'min': min_val, 'step': step}

    def _slider_value(self, sdict) -> float:
        return sdict['min'] + sdict['slider'].value() * sdict['step']

    # ---------- Actions ----------
    def _apply_velocity(self):
        new_vx = self._slider_value(self.vel_x)
        new_vy = self._slider_value(self.vel_y)
        vz = float(self.objects['platform_controller'].velocity[2])  # keep Z unchanged
        self.objects['platform_controller'].velocity = np.array([new_vx, new_vy, vz])
        print(f"GUI: platform velocity = {new_vx:.2f}, {new_vy:.2f}, {vz:.2f}")

    def _apply_wind(self):
        vx = self._slider_value(self.wind_x)
        vy = self._slider_value(self.wind_y)
        self.env.enable_wind(True)
        self.env.set_wind(velocity_world=[vx, vy, 0.0])
        print(f"GUI: wind = {vx:.2f}, {vy:.2f}, {0.00}")

    def toggle_pause_resume(self):
        if self.simulation.is_loop_state('pause'):
            print("GUI: pressed resume")
            self.simulation.set_loop_state(resume=True)
        else:
            print("GUI: pressed pause")
            self.simulation.set_loop_state(pause=True)

    def _sync_pause_label(self):
        paused = self.simulation.is_loop_state('pause')
        want = "Resume" if paused else "Pause"
        if self.pause_btn.text() != want:
            self.pause_btn.setText(want)

    def _sync_land_btn(self):
        if self.simulation.orchestrator.can_land():
            self.land_btn.setStyleSheet("background:#11b; color:black; font-weight:bold;")
            self.land_btn.setEnabled(True)
        else:
            self.land_btn.setStyleSheet("background:#d00; color:white; font-weight:bold;")
            self.land_btn.setEnabled(False)

    def _on_terminate(self):
        print("GUI: pressed terminate")
        self.simulation.set_loop_state(terminate=True)
        QtCore.QTimer.singleShot(200, QApplication.quit)

    def _on_land(self):
        print("GUI: pressed land")
        self.objects['quadrotor_controller'].descend()

    # Ensure camera thread halts on close
    def closeEvent(self, event):
        try:
            self.camera_streamer.stop()
        finally:
            super().closeEvent(event)
