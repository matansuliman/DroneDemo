from noises import GPSNoise

class BasicSensor:
    def __init__(self, env, body_id: int):
        self._env = env
        self._body_id = body_id
    
    @property
    def body_id(self):
        return self._body_id
    
    def get_pos(self, mode='noise'):
        raise NotImplementedError("Subclasses should implement this method")


class GPS(BasicSensor):
    def __init__(self, env, body_id: int):
        super().__init__(env, body_id)
        self._noise = GPSNoise(self._env) # Initialize GPS noise model

    def get_pos(self, mode='noise'):
        if mode == 'noise':
            offset, scale = self._noise.step()
            return (self.get_pos(mode='no_noise') + offset) * scale
        
        elif mode == 'no_noise':
            return self._env.world_pos_of_body(self._body_id)
        
        else:
            raise ValueError(f"Unknown mode: {mode}")