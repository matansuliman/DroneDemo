import sys
import numpy as np
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import QImage, QPixmap
import pyqtgraph as pg

from PyQt5.QtWidgets import (
    QWidget, QGroupBox, QVBoxLayout, QHBoxLayout,
    QLabel, QDoubleSpinBox
)


class TargetControlGUI(QtWidgets.QWidget):
    def __init__(self, simulation, camera_streamer):
        super().__init__()
        self.env = simulation.orchestrator.env
        self.objects = simulation.orchestrator.objects
        self.simulation = simulation
        self.landing_active = False
        self.is_paused = False

        self.camera_streamer = camera_streamer
        self.camera_enabled = True

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Quadrotor Target drone")
        self.setGeometry(100, 100, 900, 600)

        layout = QtWidgets.QVBoxLayout()

        self.pos_label = QtWidgets.QLabel("Current Position: (0.00, 0.00, 0.00)")
        layout.addWidget(self.pos_label)

        self.vel_label = QtWidgets.QLabel(f"Velocity: {self.objects['platform'].velocity.tolist()}")
        layout.addWidget(self.vel_label)

        visual_layout = QtWidgets.QHBoxLayout()

        self.xy_plot = pg.PlotWidget(title="XY Plane View")
        self.xy_plot.setMouseEnabled(x=False, y=False)
        self.xy_plot.setAspectLocked(True)
        self.xy_plot.showGrid(x=True, y=True)
        self.xy_current_dot = pg.ScatterPlotItem(pen=None, brush='g', size=10)
        self.xy_platform_dot = pg.ScatterPlotItem(pen=None, brush='b', size=10)
        self.xy_plot.addItem(self.xy_current_dot)
        self.xy_plot.addItem(self.xy_platform_dot)
        self.xy_plot.setXRange
        visual_layout.addWidget(self.xy_plot)

        self.z_plot = pg.PlotWidget(title="Z Altitude (1D Only)")
        self.z_plot.setMouseEnabled(x=False, y=False)
        self.z_plot.showGrid(x=False, y=True)
        self.z_plot.setYRange(-2, 10)
        self.z_plot.setLimits(yMin=-2)
        self.z_plot.showAxis('left')
        self.z_plot.getAxis('left').setLabel('Altitude (m)')

        self.z_current_line = pg.InfiniteLine(pos=0, angle=0, pen='g')
        self.z_platform_line = pg.InfiniteLine(pos=0, angle=0, pen='b')
        self.z_plot.addItem(self.z_current_line)
        self.z_plot.addItem(self.z_platform_line)
        visual_layout.addWidget(self.z_plot)

        layout.addLayout(visual_layout)

        self.vel_x = self._create_slider(-5.0, 5.0, self.objects['platform'].velocity[0],
                                         "Velocity X", self.update_velocity, step=0.1)
        layout.addLayout(self.vel_x['layout'])

        self.vel_y = self._create_slider(-5.0, 5.0, self.objects['platform'].velocity[1],
                                         "Velocity Y", self.update_velocity, step=0.1)
        layout.addLayout(self.vel_y['layout'])

        # --- Buttons row ---
        btn_layout = QtWidgets.QHBoxLayout()

        # Single Pause/Resume toggle
        self.pause_btn = QtWidgets.QPushButton("Pause")
        self.pause_btn.setMinimumWidth(90)
        self.pause_btn.clicked.connect(self.toggle_pause_resume)
        btn_layout.addWidget(self.pause_btn)

        # Terminate
        self.terminate_btn = QtWidgets.QPushButton("Terminate")
        self.terminate_btn.setMinimumWidth(90)
        self.terminate_btn.clicked.connect(self.on_terminate)
        btn_layout.addWidget(self.terminate_btn)

        # Land: fixed slot + styling (no layout shift)
        self.land_button = QtWidgets.QPushButton("Land")
        self.land_button.setFixedWidth(90)
        self.land_button.setStyleSheet("background-color: red; color: white; font-weight: bold;")
        self.land_button.setEnabled(False)
        self.land_button.clicked.connect(self.on_land)
        btn_layout.addWidget(self.land_button)

        btn_layout.addStretch(1)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

        # --- Timers / Camera ---
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_target)
        self.timer.timeout.connect(self.check_simulation_status)
        self.timer.timeout.connect(self._sync_pause_button_label)  # keep label in sync
        self.timer.start(100)

        self.camera_label = QtWidgets.QLabel()
        self.camera_label.setFixedSize(320, 240)
        layout.addWidget(self.camera_label)

        self.marker_detection_label = QtWidgets.QLabel("markers detection:")
        layout.addWidget(self.marker_detection_label)

        # --- Wind controls (X/Y only) ---
        wind_box = QGroupBox("Wind (X/Y)")
        wind_layout = QVBoxLayout(wind_box)

        # Vx
        row_x = QHBoxLayout()
        row_x.addWidget(QLabel("Vx (m/s):"))
        self.spin_vx = QDoubleSpinBox()
        self.spin_vx.setRange(-10.0, 10.0)
        self.spin_vx.setDecimals(2)
        self.spin_vx.setSingleStep(0.1)
        self.spin_vx.setValue(0.0)
        row_x.addWidget(self.spin_vx)
        wind_layout.addLayout(row_x)

        # Vy
        row_y = QHBoxLayout()
        row_y.addWidget(QLabel("Vy (m/s):"))
        self.spin_vy = QDoubleSpinBox()
        self.spin_vy.setRange(-50.0, 50.0)
        self.spin_vy.setDecimals(2)
        self.spin_vy.setSingleStep(0.1)
        self.spin_vy.setValue(0.0)
        row_y.addWidget(self.spin_vy)
        wind_layout.addLayout(row_y)

        # Wire up changes
        self.spin_vx.valueChanged.connect(self._on_wind_changed)
        self.spin_vy.valueChanged.connect(self._on_wind_changed)

        # Add box to your sidebar / root layout
        layout.addWidget(wind_box)

    def update_marker_detection(self, result: dict):
        if result is not None:
            self.marker_detection_label.setText(str(result))

    def update_camera_view(self, frame):
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_img = QImage(frame.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888)
        self.camera_label.setPixmap(QPixmap.fromImage(qt_img))

    def _create_slider(self, min_val, max_val, init_val, label, on_change=None, step=0.1):
        slider_layout = QtWidgets.QVBoxLayout()
        label_widget = QtWidgets.QLabel(f"{label}: {init_val:.2f}")
        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(int((max_val - min_val) / step))
        slider.setValue(int((init_val - min_val) / step))

        def update_label():
            value = min_val + (slider.value()) * step
            label_widget.setText(f"{label}: {value:.2f}")
            if on_change:
                on_change()

        slider.valueChanged.connect(update_label)
        slider_layout.addWidget(label_widget)
        slider_layout.addWidget(slider)
        return {'layout': slider_layout, 'slider': slider, 'label': label_widget, 'min': min_val, 'step': step}

    def update_velocity(self):
        new_vx = self._slider_val(self.vel_x)
        new_vy = self._slider_val(self.vel_y)
        new_vz = self.objects['platform'].velocity[2]  # Keep Z velocity unchanged
        self.objects['platform'].velocity = np.array([new_vx, new_vy, new_vz])
        self.vel_label.setText(f"Velocity: {[round(new_vx, 2), round(new_vy, 2), round(new_vz, 2)]}")

    def _slider_val(self, slider_dict):
        return slider_dict['min'] + slider_dict['slider'].value() * slider_dict['step']

    def update_target(self):
        if self.objects['drone_controller'].log['x']:
            pos = self.objects['drone'].getPos(mode='no_noise')
            self.pos_label.setText(f"Current Position: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
            self.xy_current_dot.setData([{'pos': [pos[0], pos[1]]}])
            self.z_current_line.setValue(pos[2])

            # Land button readiness logic
            error_xy = np.linalg.norm(self.objects['drone_controller'].error_xy_from_target())
            target_adjusted = getattr(self.simulation.orchestrator, "target_adjusted", True)
            if (not self.landing_active) and target_adjusted and (error_xy < 1):
                self.land_button.setStyleSheet("background-color: green; color: black; font-weight: bold;")
                self.land_button.setEnabled(True)
            else:
                self.land_button.setStyleSheet("background-color: red; color: white; font-weight: bold;")
                self.land_button.setEnabled(False)

            if self.landing_active:
                self.objects['drone_controller'].update_target(mode='landing')
        else:
            pos = self.objects['drone'].getPos(mode='no_noise')

        # Platform visuals
        px, py, pz = self.objects['platform'].getPos(mode='no_noise')
        self.xy_platform_dot.setData([{'pos': [px, py]}])
        self.z_platform_line.setValue(pz)

        # Autoscale plots around pos/target/platform
        tx, ty, tz = self.objects['drone_controller'].target
        margin = 2.0
        min_x, max_x = min(pos[0], tx, px), max(pos[0], tx, px)
        min_y, max_y = min(pos[1], ty, py), max(pos[1], ty, py)
        center_x, center_y = 0.5 * (min_x + max_x), 0.5 * (min_y + max_y)
        span_x, span_y = max_x - min_x, max_y - min_y
        half_width, half_height = 0.5 * max(span_x, 2 * margin), 0.5 * max(span_y, 2 * margin)
        self.xy_plot.setXRange(center_x - half_width, center_x + half_width, padding=0)
        self.xy_plot.setYRange(center_y - half_height, center_y + half_height, padding=0)

        min_z, max_z = min(pos[2], tz, pz), max(pos[2], tz, pz)
        self.z_plot.setYRange(min_z - margin, max_z + margin, padding=0)

    # ---- Pause/Resume toggle ----
    def toggle_pause_resume(self):
        if self.simulation.isLoopState('pause'):
            self.simulation.setLoopState(resume=True)
            self.pause_btn.setText("Pause")
            self.is_paused = False
        else:
            self.simulation.setLoopState(pause=True)
            self.pause_btn.setText("Resume")
            self.is_paused = True

    def _sync_pause_button_label(self):
        paused = self.simulation.isLoopState('pause')
        if paused != self.is_paused:
            self.is_paused = paused
            self.pause_btn.setText("Resume" if paused else "Pause")

    # ---- Other controls ----
    def on_terminate(self):
        self.simulation.setLoopState(terminate=True)
        QtCore.QTimer.singleShot(200, QtWidgets.QApplication.quit)

    def on_land(self):
        self.landing_active = True
        self.land_button.setEnabled(False)

    def check_simulation_status(self):
        if self.simulation.isLoopState('terminate'):
            self.close()

    def closeEvent(self, event):
        self.camera_streamer.stop()  # guarantee the thread halts
        super().closeEvent(event)

    def _on_wind_changed(self, *_):
        vx = float(self.spin_vx.value())
        vy = float(self.spin_vy.value())
        self.env.enable_wind(True)
        self.env.set_wind(velocity_world=[vx, vy, 0.0])


def launch_target_slider(drone, platform):
    app = QtWidgets.QApplication(sys.argv)
    gui = TargetControlGUI(drone, platform)
    gui.show()
    sys.exit(app.exec_())
