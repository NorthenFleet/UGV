import numpy as np

class SimpleVecEnv:
    def __init__(self, make_env, num_envs):
        self.envs = [make_env() for _ in range(num_envs)]
        self.num_envs = num_envs
        self.action_dim = self.envs[0].action_dim

    def reset(self):
        return np.stack([e.reset() for e in self.envs], axis=0)

    def step(self, actions):
        obs = []
        rews = []
        dones = []
        infos = []
        for i, e in enumerate(self.envs):
            o, r, d, info = e.step(actions[i])
            if d:
                o = e.reset()
            obs.append(o)
            rews.append(r)
            dones.append(d)
            infos.append(info)
        return np.stack(obs, axis=0), np.array(rews, dtype=np.float32), np.array(dones, dtype=np.bool_), infos

    def close(self):
        for e in self.envs:
            try:
                e.close()
            except Exception:
                pass