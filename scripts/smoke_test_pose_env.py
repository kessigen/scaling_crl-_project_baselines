import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from envs.wx250_pick_env import WX250PickPlaceEnv

env = WX250PickPlaceEnv(
    render_mode="human",
    asset_root=REPO_ROOT / "assets" / "external" / "interbotix_ros_manipulators" / "interbotix_ros_xsarms" / "interbotix_xsarm_descriptions" / "meshes",
)

obs, info = env.reset()

try:
    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        time.sleep(0.03)

        if terminated or truncated:
            obs, info = env.reset()
except KeyboardInterrupt:
    pass

env.close()
