import sys
import numpy as np
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import QImage, QPixmap
import pyqtgraph as pg


class TargetControlGUI(QtWidgets.QWidget):
    def __init__(self, orchestrator, camera_streamer):
        
        super().__init__()
        self.objects = orchestrator._objects
        self.orchestrator = orchestrator
        self.landing_active = False

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
        #self.xy_target_dot = pg.ScatterPlotItem(pen=None, brush='r', size=10)
        self.xy_platform_dot = pg.ScatterPlotItem(pen=None, brush='b', size=10)
        self.xy_plot.addItem(self.xy_current_dot)
        #self.xy_plot.addItem(self.xy_target_dot)
        self.xy_plot.addItem(self.xy_platform_dot)
        visual_layout.addWidget(self.xy_plot)

        self.z_plot = pg.PlotWidget(title="Z Altitude (1D Only)")
        self.z_plot.setMouseEnabled(x=False, y=False)
        self.z_plot.showGrid(x=False, y=True)
        self.z_plot.setYRange(-2, 10)
        self.z_plot.setLimits(yMin=-2)
        self.z_plot.showAxis('left')
        self.z_plot.getAxis('left').setLabel('Altitude (m)')

        self.z_current_line = pg.InfiniteLine(pos=0, angle=0, pen='g')
        #self.z_target_line = pg.InfiniteLine(pos=0, angle=0, pen='r')
        self.z_platform_line = pg.InfiniteLine(pos=0, angle=0, pen='b')
        self.z_plot.addItem(self.z_current_line)
        #self.z_plot.addItem(self.z_target_line)
        self.z_plot.addItem(self.z_platform_line)
        visual_layout.addWidget(self.z_plot)

        layout.addLayout(visual_layout)

        self.vel_x = self._create_slider(-5.0, 5.0, self.objects['platform'].velocity[0], "Velocity X", self.update_velocity, step=0.1)
        layout.addLayout(self.vel_x['layout'])

        self.vel_y = self._create_slider(-5.0, 5.0, self.objects['platform'].velocity[1], "Velocity Y", self.update_velocity, step=0.1)
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

        self.camera_label = QtWidgets.QLabel()
        self.camera_label.setFixedSize(320, 240)
        layout.addWidget(self.camera_label)

        self.marker_detection_label = QtWidgets.QLabel("markers detection:")
        layout.addWidget(self.marker_detection_label)

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
            pos = (
                self.objects['drone_controller'].log['x'][-1],
                self.objects['drone_controller'].log['y'][-1],
                self.objects['drone_controller'].log['z'][-1]
            )
            self.pos_label.setText(f"Current Position: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
            self.xy_current_dot.setData([{'pos': [pos[0], pos[1]]}])
            self.z_current_line.setValue(pos[2])

            tx, ty, tz = self.objects['drone_controller'].target
            error_xy = np.linalg.norm(np.array([tx, ty]) - np.array([pos[0], pos[1]]))

            self.land_button.setVisible(not self.landing_active and error_xy < 1)

            if self.landing_active:
                self.objects['drone_controller'].update_target(mode='landing')
        else:
            pos = self.objects['drone'].getPos(mode='no_noise')

        tx, ty, tz = self.objects['drone_controller'].target

        px, py, pz = self.objects['platform'].getPos(mode='no_noise')
        self.xy_platform_dot.setData([{'pos': [px, py]}])
        self.z_platform_line.setValue(pz)

        margin = 2.0
        min_x = min(pos[0], tx, px)
        max_x = max(pos[0], tx, px)
        min_y = min(pos[1], ty, py)
        max_y = max(pos[1], ty, py)

        center_x = 0.5 * (min_x + max_x)
        center_y = 0.5 * (min_y + max_y)
        span_x = max_x - min_x
        span_y = max_y - min_y
        half_width = 0.5 * max(span_x, 2 * margin)
        half_height = 0.5 * max(span_y, 2 * margin)

        self.xy_plot.setXRange(center_x - half_width, center_x + half_width, padding=0)
        self.xy_plot.setYRange(center_y - half_height, center_y + half_height, padding=0)

        min_z = min(pos[2], tz, pz)
        max_z = max(pos[2], tz, pz)
        self.z_plot.setYRange(min_z - margin, max_z + margin, padding=0)

    def on_pause(self):
        self.orchestrator.ChangeLoopState(pause=True)

    def on_resume(self):
        self.orchestrator.ChangeLoopState(resume=True)

    def on_terminate(self):
        self.orchestrator.ChangeLoopState(terminate=True)
        QtCore.QTimer.singleShot(200, QtWidgets.QApplication.quit)

    def on_land(self):
        self.landing_active = True
        self.land_button.setVisible(False)

    def check_simulation_status(self):
        if self.orchestrator.isLoopState('terminate'):
            self.close()

    def closeEvent(self, event):
        self.camera_streamer.stop()     # guarantee the thread halts
        super().closeEvent(event)

def launch_target_slider(drone, platform):
    app = QtWidgets.QApplication(sys.argv)
    gui = TargetControlGUI(drone, platform)
    gui.show()
    sys.exit(app.exec_())