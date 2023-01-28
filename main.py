import numpy as np

from src.environment.draw import draw_instance
from src.environment.gym import ENV_DIR, EternityEnv

env = EternityEnv(ENV_DIR / "eternity_A.txt", manual_orient=False)
env.reset()

draw_instance(env.instance, "frame-1.png")

env.step(np.array([0, 3]))
draw_instance(env.instance, "frame-2.png")
