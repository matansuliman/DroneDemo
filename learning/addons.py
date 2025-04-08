"""
# # for sweeping
# file: target_slider_gui.py (full version with manual XY control and sweep toggle)
from simple_pid import PID
import sys
import numpy as np
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import QImage, QPixmap
import pyqtgraph as pg
import threading
import time

class TargetControlGUI(QtWidgets.QWidget):
    def __init__(self, drone, platform, camera_streamer):
        super().__init__()
        self.drone = drone
        self.platform = platform
        self.landing_active = False

        self.camera_streamer = camera_streamer
        self.camera_enabled = True

        self.manual_xy_target = np.array([0.0, 0.0])
        self.sweep_running = True

        self.init_ui()

        self.auto_thread = threading.Thread(target=self.auto_sweep_xy, daemon=True)
        self.auto_thread.start()

    def toggle_sweep(self):
        self.sweep_running = not self.sweep_running
        self.sweep_toggle_btn.setText("Pause Sweep" if self.sweep_running else "Resume Sweep")

    def auto_sweep_xy(self):
        direction = 1
        for y in np.arange(-4, 4.01, 0.5):
            x_range = np.arange(-3, 3.01, 0.2) if direction == 1 else np.arange(3, -3.01, -0.2)
            for x in x_range:
                while not self.sweep_running:
                    time.sleep(0.05)
                self.manual_xy_target = np.array([x, y])
                self.sweep_label.setText(f"Sweep: Y={y:.1f}, X={x:.1f}")
                time.sleep(1)
            time.sleep(1)
            direction *= -1
        self.sweep_label.setText("Sweep: done")

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

    def _slider_val(self, slider_dict):
        return slider_dict['min'] + slider_dict['slider'].value() * slider_dict['step']

    def update_velocity(self):
        vx = self._slider_val(self.vel_x)
        vy = self._slider_val(self.vel_y)
        self.platform.velocity = np.array([vx, vy, self.platform.velocity[2]])
        self.vel_label.setText(f"Velocity: {[round(vx, 2), round(vy, 2), round(self.platform.velocity[2], 2)]}")

    def toggle_camera(self):
        if not self.camera_streamer:
            return
        if self.camera_enabled:
            self.camera_streamer.stop()
            self.toggle_camera_btn.setText("Start Camera")
        else:
            self.camera_streamer.start()
            self.toggle_camera_btn.setText("Stop Camera")
        self.camera_enabled = not self.camera_enabled

    def update_camera_view(self, frame):
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_img = QImage(frame.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888)
        self.camera_label.setPixmap(QPixmap.fromImage(qt_img))

    def init_ui(self):
        self.setWindowTitle("Quadrotor Target drone")
        self.setGeometry(100, 100, 900, 600)

        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)

        self.pos_label = QtWidgets.QLabel("Current Position: (0.00, 0.00, 0.00)")
        layout.addWidget(self.pos_label)

        self.vel_label = QtWidgets.QLabel(f"Velocity: {self.platform.velocity.tolist()}")
        layout.addWidget(self.vel_label)

        visual_layout = QtWidgets.QHBoxLayout()
        self.xy_plot = pg.PlotWidget(title="XY Plane View")
        self.xy_plot.setMouseEnabled(x=False, y=False)
        self.xy_plot.setAspectLocked(True)
        self.xy_plot.showGrid(x=True, y=True)
        self.xy_current_dot = pg.ScatterPlotItem(pen=None, brush='g', size=10)
        self.xy_target_dot = pg.ScatterPlotItem(pen=None, brush='r', size=10)
        self.xy_plot.addItem(self.xy_current_dot)
        self.xy_plot.addItem(self.xy_target_dot)
        visual_layout.addWidget(self.xy_plot)
        layout.addLayout(visual_layout)

        self.vel_x = self._create_slider(-5.0, 5.0, self.platform.velocity[0], "Velocity X", self.update_velocity, step=0.1)
        layout.addLayout(self.vel_x['layout'])
        self.vel_y = self._create_slider(-5.0, 5.0, self.platform.velocity[1], "Velocity Y", self.update_velocity, step=0.1)
        layout.addLayout(self.vel_y['layout'])
        self.pos_x = self._create_slider(-5.0, 5.0, 0.0, "Target X", self.update_target_from_sliders, step=0.1)
        layout.addLayout(self.pos_x['layout'])
        self.pos_y = self._create_slider(-5.0, 5.0, 0.0, "Target Y", self.update_target_from_sliders, step=0.1)
        layout.addLayout(self.pos_y['layout'])

        self.sweep_label = QtWidgets.QLabel("Sweep: idle")
        layout.addWidget(self.sweep_label)

        self.sweep_toggle_btn = QtWidgets.QPushButton("Pause Sweep")
        self.sweep_toggle_btn.clicked.connect(self.toggle_sweep)
        layout.addWidget(self.sweep_toggle_btn)

        btn_layout = QtWidgets.QHBoxLayout()
        for text, handler in [("Pause", self.on_pause), ("Resume", self.on_resume), ("Terminate", self.on_terminate)]:
            btn = QtWidgets.QPushButton(text)
            btn.clicked.connect(handler)
            btn_layout.addWidget(btn)

        self.land_button = QtWidgets.QPushButton("Land")
        self.land_button.clicked.connect(self.on_land)
        self.land_button.setVisible(False)
        btn_layout.addWidget(self.land_button)

        self.toggle_camera_btn = QtWidgets.QPushButton("Toggle Camera")
        self.toggle_camera_btn.clicked.connect(self.toggle_camera)
        btn_layout.addWidget(self.toggle_camera_btn)
        layout.addLayout(btn_layout)

        self.camera_label = QtWidgets.QLabel()
        self.camera_label.setFixedSize(320, 240)
        layout.addWidget(self.camera_label)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_target)
        self.timer.timeout.connect(self.check_simulation_status)
        self.timer.start(100)

    def update_target_from_sliders(self):
        x = self._slider_val(self.pos_x)
        y = self._slider_val(self.pos_y)
        self.manual_xy_target = np.array([x, y])

    def update_target(self):
        x, y = self.manual_xy_target
        self.drone.target[0] = x
        self.drone.target[1] = y
        self.drone.pid_x.setpoint = x
        self.drone.pid_y.setpoint = y

        if self.drone.log['x']:
            pos = (
                self.drone.log['x'][-1],
                self.drone.log['y'][-1],
                self.drone.log['z'][-1]
            )
            self.pos_label.setText(f"Current Position: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
            self.xy_current_dot.setData([{'pos': [pos[0], pos[1]]}])
            tx, ty = self.manual_xy_target
            error_xy = np.linalg.norm(np.array([tx, ty]) - np.array([pos[0], pos[1]]))
            self.land_button.setVisible(not self.landing_active and error_xy < 0.1)
            if self.landing_active:
                self.drone.target_z_ff = 0.1
                self.drone.pid_z.output_limits = (-0.20, 0.10)
        tx, ty = self.manual_xy_target
        self.xy_target_dot.setData([{'pos': [tx, ty]}])
        self.xy_plot.setXRange(tx - 2, tx + 2)
        self.xy_plot.setYRange(ty - 2, ty + 2)

    def on_pause(self):
        self.drone.paused = True

    def on_resume(self):
        self.drone.paused = False

    def on_terminate(self):
        self.drone.terminated = True
        QtCore.QTimer.singleShot(200, QtWidgets.QApplication.quit)

    def on_land(self):
        self.landing_active = True
        self.land_button.setVisible(False)

    def check_simulation_status(self):
        if self.drone.terminated:
            self.close()


def launch_target_slider(drone, platform):
    app = QtWidgets.QApplication(sys.argv)
    gui = TargetControlGUI(drone, platform)
    gui.show()
    sys.exit(app.exec_())


"""