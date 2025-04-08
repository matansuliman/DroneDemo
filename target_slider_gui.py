# file: target_slider_gui.py
# Reduced GUI: Only velocity X and Y sliders remain, 0.1 step size

from simple_pid import PID
import sys
import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg

class TargetControlGUI(QtWidgets.QWidget):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.landing_active = False
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Quadrotor Target Controller")
        self.setGeometry(100, 100, 900, 600)

        layout = QtWidgets.QVBoxLayout()

        self.pos_label = QtWidgets.QLabel("Current Position: (0.00, 0.00, 0.00)")
        layout.addWidget(self.pos_label)

        self.vel_label = QtWidgets.QLabel(f"Velocity: {self.controller.velocity.tolist()}")
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

        self.z_plot = pg.PlotWidget(title="Z Altitude (1D Only)")
        self.z_plot.setMouseEnabled(x=False, y=False)
        self.z_plot.showGrid(x=False, y=True)
        self.z_plot.setXRange(0, 1)
        self.z_plot.setYRange(0, 10)
        self.z_plot.hideAxis('bottom')

        self.z_current_line = pg.InfiniteLine(pos=0, angle=0, pen='g')
        self.z_target_line = pg.InfiniteLine(pos=0, angle=0, pen='r')
        self.z_plot.addItem(self.z_current_line)
        self.z_plot.addItem(self.z_target_line)
        visual_layout.addWidget(self.z_plot)

        layout.addLayout(visual_layout)

        self.vel_x = self._create_slider(-5.0, 5.0, self.controller.velocity[0], "Velocity X", self.update_velocity, step=0.1)
        layout.addLayout(self.vel_x['layout'])

        self.vel_y = self._create_slider(-5.0, 5.0, self.controller.velocity[1], "Velocity Y", self.update_velocity, step=0.1)
        layout.addLayout(self.vel_y['layout'])

        btn_layout = QtWidgets.QHBoxLayout()
        for text, handler in [("Pause", self.on_pause), ("Resume", self.on_resume), ("Terminate", self.on_terminate)]:
            btn = QtWidgets.QPushButton(text)
            btn.clicked.connect(handler)
            btn_layout.addWidget(btn)

        self.land_button = QtWidgets.QPushButton("Land")
        self.land_button.clicked.connect(self.on_land)
        self.land_button.setVisible(False)
        btn_layout.addWidget(self.land_button)

        layout.addLayout(btn_layout)
        self.setLayout(layout)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_target)
        self.timer.timeout.connect(self.check_simulation_status)
        self.timer.start(100)

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
        vx = self._slider_val(self.vel_x)
        vy = self._slider_val(self.vel_y)
        self.controller.velocity = np.array([vx, vy, self.controller.velocity[2]])
        self.vel_label.setText(f"Velocity: {[round(vx,2), round(vy,2), round(self.controller.velocity[2],2)]}")

    def _slider_val(self, slider_dict):
        return slider_dict['min'] + slider_dict['slider'].value() * slider_dict['step']

    def update_target(self):
        if self.controller.log['x']:
            pos = (
                self.controller.log['x'][-1],
                self.controller.log['y'][-1],
                self.controller.log['z'][-1]
            )
            self.pos_label.setText(f"Current Position: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
            self.xy_current_dot.setData([{'pos': [pos[0], pos[1]]}])
            self.z_current_line.setValue(pos[2])

            tx, ty, tz = self.controller.target
            error_xy = np.linalg.norm(np.array([tx, ty]) - np.array([pos[0], pos[1]]))

            # Show land button if drone is over platform and close in Z
            self.land_button.setVisible(not self.landing_active and error_xy < 0.1)

            # Gradually descend during landing
            if self.landing_active:
                self.controller.target_z_ff = 0.1
                self.controller.pid_z.output_limits = (-0.20, 0.10)

        tx, ty, tz = self.controller.target
        self.xy_target_dot.setData([{'pos': [tx, ty]}])
        self.z_target_line.setValue(tz)
        self.xy_plot.setXRange(tx - 2, tx + 2)
        self.xy_plot.setYRange(ty - 2, ty + 2)

    def on_pause(self):
        self.controller.paused = True

    def on_resume(self):
        self.controller.paused = False

    def on_terminate(self):
        self.controller.terminated = True
        QtCore.QTimer.singleShot(200, QtWidgets.QApplication.quit)

    def on_land(self):
        self.landing_active = True
        self.land_button.setVisible(False)

    def check_simulation_status(self):
        if self.controller.terminated:
            self.close()

def launch_target_slider(controller):
    app = QtWidgets.QApplication(sys.argv)
    gui = TargetControlGUI(controller)
    gui.show()
    sys.exit(app.exec_())