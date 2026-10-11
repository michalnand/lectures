import numpy
import envpool
import cv2 
 
class VecEnv: 
    def __init__(self, env_name="LunarLander-v3", n_envs=16, render_width=512, render_height=512):
        self.n_envs = n_envs
        self.env = envpool.make_gymnasium(
            env_name,
            num_envs=n_envs,
            render_mode="rgb_array",
            render_width=render_width,
            render_height=render_height,
        )
        self.observation_space = self.env.observation_space   # Box(8,)
        self.action_space = self.env.action_space             # Discrete(4)
 
    def reset(self, env_idx=None):
        if env_idx is None:
            state, _ = self.env.reset()
            return state
            
        env_ids = numpy.array([env_idx], dtype=numpy.int32)
        state, _ = self.env.reset(env_ids)
        
        return state[0]
 
    def step(self, actions):
        state, reward, term, trunc, _ = self.env.step(actions)

        
        return state, reward/100.0, numpy.logical_or(term, trunc)
 
    def render(self, env_id=0, wait_ms=1):
        frame = self.env.render(env_ids=[env_id])[0]
        cv2.imshow("VecEnv", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        return cv2.waitKey(wait_ms) & 0xFF != 27