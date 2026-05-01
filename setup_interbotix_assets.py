from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

REPO_URL = "https://github.com/Interbotix/interbotix_ros_manipulators.git"
REL_MESH_DIR = Path("interbotix_ros_xsarms/interbotix_xsarm_descriptions/meshes")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clone the public Interbotix descriptions repo so the wx250 mesh assets are available locally.")
    parser.add_argument("--target", default="assets/external/interbotix_ros_manipulators", help="Clone destination")
    args = parser.parse_args()

    target = Path(args.target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(target)], check=True)

    mesh_dir = target / REL_MESH_DIR
    print(f"Interbotix repo: {target.resolve()}")
    print(f"Mesh root to pass into the env: {mesh_dir.resolve()}")
    print("Expected files include interbotix_black.png and wx250_meshes/wx250_1_base.stl")


if __name__ == "__main__":
    main()
