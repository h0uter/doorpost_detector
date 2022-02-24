import os
import numpy as np
import doorpost_detector.api as dpd
from doorpost_detector.utils.viz_lvl import VizLVL


def demo_with_full_viz():
    path = os.path.abspath(".")
    points = np.load(f"{path}/data/door1_cropped_m0_8.npy")

    response = dpd.doorpost_pose_from_cropped_pointcloud_usecase(points, vis=VizLVL.EVERY_STEP)
    print(
        f">>> dataset 1: success: {response.success}, poses: {response.poses}"
    )


if __name__ == "__main__":
    demo_with_full_viz()