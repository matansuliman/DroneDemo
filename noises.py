import numpy as np

DEFAULT_BIAS_STD = 2
DEFAULT_DRIFT_RATE_STD = 0.01
DEFAULT_SCALE_NOISE_STD = 0.5

Z_BIAS_SCALE = 0.2


class GPSNoise:
    """
    GPSNoiseModel simulates GPS noise with:
    - static bias (fixed offset)
    - drift (accumulating over time)
    - multiplicative Gaussian noise (scaling factor)
    """

    def __init__(self, 
                 bias_stddev: float = DEFAULT_BIAS_STD, 
                 drift_rate_stddev: float = DEFAULT_DRIFT_RATE_STD, 
                 scale_noise_stddev: float = DEFAULT_SCALE_NOISE_STD):
        self.bias = np.random.normal(0, bias_stddev, size=3)
        self.drift_rate = np.random.normal(0, drift_rate_stddev, size=3)
        self.drift = np.zeros(3)
        self.scale_noise_stddev = scale_noise_stddev

        self.bias[2] *= Z_BIAS_SCALE

    def step(self, true_position: np.ndarray, dt) -> np.ndarray:
        self.drift += self.drift_rate * dt
        offset = self.bias + self.drift
        scale = np.random.normal(1.0, self.scale_noise_stddev, size=3)
        return (true_position + offset) * scale

    def reset(self):
        self.drift = np.zeros(3)
