import socket
import json
import time

class UDPSender:
    def __init__(self, ip: str, port: int, src: str = "sim"):
        self.addr = (ip, int(port))
        self.src = src
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.joint_order = [
            "LF_HIP", "LF_THIGH", "LF_KNEE",
            "LM_HIP", "LM_THIGH", "LM_KNEE",
            "LR_HIP", "LR_THIGH", "LR_KNEE",
            "RF_HIP", "RF_THIGH", "RF_KNEE",
            "RM_HIP", "RM_THIGH", "RM_KNEE",
            "RR_HIP", "RR_THIGH", "RR_KNEE",
        ]

    def send(self, obs: dict):
        msg = {
            "src": self.src,
            "timestamp": time.time(),
            "base": {"pos": list(map(float, obs.get("base_pos", [0, 0, 0]))), "ori": list(map(float, obs.get("base_ori", [0, 0, 0, 1])))},
            "joints": {"order": self.joint_order, "angles": [float(x) for x in obs.get("q", [])]},
            "extra": {"contact": [float(x) for x in obs.get("contacts", [])]},
        }
        data = json.dumps(msg).encode("utf-8")
        self.sock.sendto(data, self.addr)

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass