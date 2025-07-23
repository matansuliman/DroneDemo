import mujoco.viewer

DISTANCE_FROM_PLATFORM = 0.01
CAMERA_DISTANCE_MULTIPLIER = 1.3

CAMERA_DISTANCE_OFFSET = 4

class Orchestrator:
    def __init__(self, env, objects):

        self._env = env
        self._objects = objects

        self._viewer = mujoco.viewer.launch_passive(env.model, env.data)

        self._loop_terminated = False
        self._loop_paused = False
        self._drone_locked = False

        self._rel_endx = None
        self._rel_endy = None
    
    @property
    def env(self):
        return self._env
    
    @property
    def objects(self):
        return self._objects

    @property
    def loop_paused(self):
        return self._loop_paused

    def pause(self):
        self._loop_paused = True

    def resume(self):
        self._loop_paused = False

    @property
    def loop_terminated(self):
        return self._loop_terminated

    def terminate(self):
        self._loop_terminated = True

    @property
    def drone_locked(self):
        return self._drone_locked
    
    def lock(self):
        self._drone_locked = True

    def unlock(self):
        self._drone_locked = False


    def mujuco_step(self):
        mujoco.mj_step(self._env.model, self._env.data)

    def _drone_is_close_to_platform(self):
        dronePos = self._objects['drone'].getTruePos()
        platformPos = self._objects['platform'].getTruePos()

        if dronePos[2] - platformPos[2] < DISTANCE_FROM_PLATFORM:
            self._rel_endx = dronePos[0] - platformPos[0]
            self._rel_endy = dronePos[1] - platformPos[1]
            return True
        return False 

    def _lock_drone_to_platform(self):
        self._env.data.qpos[:3] = self._objects['platform'].getTruePos()[:3] + [self._rel_endx, self._rel_endy, DISTANCE_FROM_PLATFORM]
        self._env.data.qvel[:] = 0
        self._env.data.qacc[:] = 0
        self._env.data.qpos[3:7] = [1, 0, 0, 0]  # upright orientation

    def update_platform(self):
        self._objects['platform_controller'].step()

    def update_drone(self):
        if self.drone_locked:
            self._lock_drone_to_platform()

        elif self._drone_is_close_to_platform():
            self.lock()
        
        else:
            self._objects['drone_controller'].update_target(self._objects['platform'].getTruePos(), 
                                                                            self._objects['platform'].velocity)
            self._objects['drone_controller'].step()

    def _update_camera_view(self):
        dronePos = self._objects['drone'].getTruePos()
        platformPos = self._objects['platform'].getTruePos()

        self._viewer.cam.distance = CAMERA_DISTANCE_MULTIPLIER * abs(dronePos[2] - platformPos[2]) + CAMERA_DISTANCE_OFFSET
        self._viewer.cam.lookat[:] = (dronePos[:3] + platformPos[:3]) / 2

    def update_scene(self):
        self.update_platform()
        self.update_drone()
        self._update_camera_view()
        self._viewer.sync()
