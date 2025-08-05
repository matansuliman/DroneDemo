import time

start = time.time()
import mujoco.viewer

import numpy as np

from models import Drone, MovingPlatform
from controllers import QuadrotorController, MovingPlatformController

from predictors import MarkerDetector

from environment import ENV

PAUSE_SLEEP_SEC = 0.1

PATH_TO_XML = "skydio_x2/scene.xml"

DISTANCE_FROM_PLATFORM = 0.01
CAMERA_DISTANCE_MULTIPLIER = 1.3

CAMERA_DISTANCE_OFFSET = 4

class basicOrchestrator:
    def __init__(self):

        #init environment
        model = mujoco.MjModel.from_xml_path(PATH_TO_XML)
        data = mujoco.MjData(model)
        dt = model.opt.timestep
        self._env = ENV(model, data, dt)

        # Initialize objects
        self._objects =  {}

        self._loop_state = 'resume'

    @property
    def env(self):
        return self._env
    
    @property
    def objects(self):
        return self._objects
    
    def ChangeLoopState(self, terminate=False, pause=False, resume=False):
        if terminate:
            self._loop_state = 'terminate'
        elif pause:
            self._loop_state = 'pause'
        elif resume:
            self._loop_state = 'resume'
        else:
            raise ValueError("Invalid loop state change request.")
        
    def isLoopState(self, state):
        return self._loop_state == state
        
    def _step_scene(self):
        raise NotImplementedError("Subclasses should implement this method")

    def loop(self): 
        raise NotImplementedError("Subclasses should implement this method")


class FollowTarget(basicOrchestrator):
    def __init__(self):
        super().__init__()

        # Initialize objects
        drone = Drone(self._env)
        platform = MovingPlatform(self._env)
        self._objects =  {
            'viewer': mujoco.viewer.launch_passive(self._env.model, self._env.data),
            'drone': drone,
            'platform': platform,
            'drone_controller': QuadrotorController(self._env, drone),
            'platform_controller': MovingPlatformController(self._env, platform)
        }

        self._locked_rel_pos_xy = None
        self._predictor = None
        self._adjust = np.array([0,0,0])
    
    @property
    def env(self):
        return self._env
    
    @property
    def objects(self):
        return self._objects

    def _drone_is_close_to_platform(self):
        dronePos = self._objects['drone'].getPos(mode='no_noise')
        platformPos = self._objects['platform'].getPos(mode='no_noise')

        if dronePos[2] - platformPos[2] < DISTANCE_FROM_PLATFORM:
            self._locked_rel_pos_xy = (dronePos[0] - platformPos[0], 
                                       dronePos[1] - platformPos[1])
            return True
        return False
        
    def _step_scene(self):
        # step platform
        self._objects['platform_controller'].step()
        
        # step drone
        if self._objects['platform_controller']._locks_activated:
            platformPos = self._objects['platform'].getPos(mode='no_noise')
            relPos = [self._locked_rel_pos_xy[0], self._locked_rel_pos_xy[1], DISTANCE_FROM_PLATFORM]
            
            self._objects['drone_controller'].update_target(mode = 'hardcode',
                                                            data = {'qpos' : platformPos + relPos})

        elif self._drone_is_close_to_platform():
            self._objects['platform_controller'].activate_locks()
        
        else:
            if ((time.time()-start > 10) and 
                (self._adjust == np.array([0,0,0])).all() and
                self._predictor is not None and
                self._predictor.is_stable()):
                    
                    self._adjust = np.append(self._predictor.get_mean_from_history() /29, 0)
            
            res = self._objects['platform'].getPos(mode='noise') - self._adjust
            self._objects['drone_controller'].update_target(mode = 'follow',
                                                            data = {'new_target_pos' : res, 
                                                                    'new_target_vel' : self._objects['platform'].velocity}
                                                            )
            self._objects['drone_controller'].step()

        # step camera view
        dronePos = self._objects['drone'].getPos(mode='no_noise')
        platformPos = self._objects['platform'].getPos(mode='no_noise')

        self._objects['viewer'].cam.distance = CAMERA_DISTANCE_MULTIPLIER * abs(dronePos[2] - platformPos[2]) + CAMERA_DISTANCE_OFFSET
        self._objects['viewer'].cam.lookat[:] = (dronePos[:3] + platformPos[:3]) / 2

        # sync viewer
        self._objects['viewer'].sync()

    def loop(self): 
        while True:
            mujoco.mj_step(self._env.model, self._env.data)
            
            if   self.isLoopState('terminate'): break
            elif self.isLoopState('pause'):     time.sleep(PAUSE_SLEEP_SEC)
            else:
                self._step_scene()
                time.sleep(self._env.dt)


class Trainer(basicOrchestrator):

    def __init__(self):
        super().__init__()
        self._t = 0.2
        self._time = time.time()
        self._step = 0.1
        self._jump = [-0.3, -0.45, 1]
        self._stop = 4

        self._start = [-3, -4.5, 10]
        self._end = [0, 0, 10]
        self._curr = self._start.copy()

    def _update_data(self):
        if self._curr[2] == self._stop:
            self.ChangeLoopState(terminate=True)
            return

        if time.time() - self._time > self._t:
            self._time = time.time()
            
            if self._curr[1] >= self._end[1]:
                self._start = [self._start[i] - self._jump[i] for i in range(3)]
                self._end = [0, 0, self._end[2] - self._jump[2]]
                self._curr = self._start.copy()
                print("jump to next row")
            
            self._curr[0] += self._step
            if self._curr[0] > self._end[0]:
                self._curr[1] += self._step
                self._curr[0] = self._start[0]

        print(f"Current position: {self._curr[0]:.2f}, {self._curr[1]:.2f}, {self._curr[2]:.2f}", end='\r')

    def _step_scene(self):
        if self._curr == self._end:
            self.ChangeLoopState(terminate=True)
        
        else:
            self._objects['drone_controller'].update_target(mode = 'hardcode', data = {'qpos' : self._curr})
            self._objects['drone_controller'].step()

            # step camera view
            dronePos = self._objects['drone'].getPos(mode='no_noise')
            platformPos = self._objects['platform'].getPos(mode='no_noise')

            self._objects['viewer'].cam.distance = CAMERA_DISTANCE_MULTIPLIER * abs(dronePos[2] - platformPos[2]) + CAMERA_DISTANCE_OFFSET
            self._objects['viewer'].cam.lookat[:] = (dronePos[:3] + platformPos[:3]) / 2

            # sync viewer
            self._objects['viewer'].sync()

    def loop(self):

        while True:
            mujoco.mj_step(self._env.model, self._env.data)
            if self.isLoopState('terminate'): break
            else: 
                self._step_scene()
                self._update_data()
                time.sleep(self._env.dt)