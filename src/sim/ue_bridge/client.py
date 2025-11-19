import numpy as np

class UEClient:
    def __init__(self):
        self.connected = False

    def connect(self):
        self.connected = True

    def reset(self, seed: int | None = None):
        return True

    def send_action(self, action: np.ndarray):
        return True

    def get_state(self) -> dict:
        return {
            "base_pos": np.zeros(3, dtype=np.float32),
            "base_ori": np.array([0, 0, 0, 1], dtype=np.float32),
            "base_euler": np.zeros(3, dtype=np.float32),
            "lin_vel": np.zeros(3, dtype=np.float32),
            "ang_vel": np.zeros(3, dtype=np.float32),
            "q": np.zeros(18, dtype=np.float32),
            "dq": np.zeros(18, dtype=np.float32),
            "contacts": np.zeros(6, dtype=np.float32),
        }

    def close(self):
        self.connected = False