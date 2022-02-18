from door_post_pose_detector.usecases.detect_door_usecase import (
    cropped_pointcloud_to_door_post_poses_usecase,
)
import numpy as np
import os

from door_post_pose_detector.utils.utils import npy2pcd



def test_run():
    # points = np.load('data/robot_cropped_pointcloud.npy')
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


        response = cropped_pointcloud_to_door_post_poses_usecase(points, vis=0)
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
    response = cropped_pointcloud_to_door_post_poses_usecase(points, vis=0)
    assert response["success"] == True


def pick_points(points: list):
    pcd = npy2pcd(points)
    print("")
    print(
        "1) Please pick at least three correspondences using [shift + left click]"
    )
    print("   Press [shift + right click] to undo point picking")
    print("2) After picking points, press 'Q' to close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    return vis.get_picked_points()



def get_ground_truth():
    # points = np.load('data/robot_cropped_pointcloud.npy')
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


        pick_points(points)
        # response = cropped_pointcloud_to_door_post_poses_usecase(points, vis=0)
        print(
            f">>> dataset {i}: success: {response['success']}, poses: {response['poses']}"
        )

if __name__ == "__main__":
    # test_pc1_success()
    # test_run()
    get_ground_truth()

