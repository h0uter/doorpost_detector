import numpy as np
import os

import doorpost_detector.api as dpd
from doorpost_detector.utils.viz_lvl import VizLVL


def test_run():
    path = os.path.abspath(".")

    for i in range(0, 7):
        points = None
        if i == 0:
            points = np.load(f"{path}/data/door0_cropped.npy")
        else:
            # crop margin 0.8m works decent
            points = np.load(f"{path}/data/door{i}_cropped_m0_8.npy")
            # crop margin 1.5m has lots of problems
            # points = np.load(f'data/door{i}_cropped_m1_5.npy')

        response = dpd.doorpost_pose_from_cropped_pointcloud_usecase(
            points, vis=VizLVL.RESULT_ONLY
        )
        print(
            f">>> dataset {i}: success: {response.success}, poses: {response.poses}"
        )


if __name__ == "__main__":
    # test_pc1_success()
    test_run()
