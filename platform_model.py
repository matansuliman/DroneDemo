import numpy as np

from sensors import GPS, INS

XML_NAME = 'platform'
XML_JOINT_NAME_X = 'platform_x'
XML_JOINT_NAME_Y = 'platform_y'

DEFAULT_VELOCITY = [0.0, 0.0, 0.0]


class MovingPlatform:
    def __init__(self,
                 model, 
                 data,
                 dt: float = 0.01, 
                 velocity=DEFAULT_VELOCITY):

        self.model = model
        self.data = data
        self.dt = dt

        # init body id
        self.body_id = next(i for i in range(self.model.nbody) if self.model.body(i).name == XML_NAME)

        # Initialize sensors
        self.gps = GPS(data, self.body_id, dt)
        self.ins = INS(data, self.body_id, dt) # internal navigation system

        self.velocity = np.array(velocity, dtype=np.float64)

        self.platform_x_id = self.model.joint(XML_JOINT_NAME_X).qposadr
        self.platform_y_id = self.model.joint(XML_JOINT_NAME_Y).qposadr

        self.log = {'time': [], 'x': [], 'y': [], 'z': []}

    def step(self):
        # Update position based on velocity
        curr_pos = self.gps.get_true_pos()
        new_pos = curr_pos + self.velocity * self.dt
        self.data.qpos[self.platform_x_id] = new_pos[0]
        self.data.qpos[self.platform_y_id] = new_pos[1]

        # Logging true position
        self.log['time'].append(self.data.time)
        pos = self.gps.get_true_pos()
        self.log['x'].append(pos[0])
        self.log['y'].append(pos[1])
        self.log['z'].append(pos[2])

    def get_vel(self):
        return self.velocity