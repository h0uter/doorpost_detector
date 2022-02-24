import numpy as np
import os

import doorpost_detector.api as dpd


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

        response = dpd.doorpost_pose_from_cropped_pointcloud_usecase(points, vis=0)
        print(
            f">>> dataset {i}: success: {response['success']}, poses: {response['poses']}"
        )


def test_pc1_success():
    path = os.path.abspath(".")
    full_path = os.path.join(path, "data", "door1_cropped_m0_8.npy")
    print(full_path)
    points = np.load(full_path)
    # points = np.load(f'{path}/data/door1_cropped_m0_8.npy')
    # points = np.load(f'../../data/door1_cropped_m0_8.npy')
    response = dpd.doorpost_pose_from_cropped_pointcloud_usecase(points, vis=0)
    assert response["success"] is True


if __name__ == "__main__":
    # test_pc1_success()
    test_run()