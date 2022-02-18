import numpy as np
import os
import open3d as o3d

from door_post_pose_detector.utils.utils import npy2pcd



def pick_points(points: list, dataset_idx):
    pcd = npy2pcd(points)
    print("")
    print(
        "1) Please pick at least three correspondences using [shift + left click]"
    )
    print("   Press [shift + right click] to undo point picking")
    print("2) After picking points, press 'Q' to close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name=f"Pick ground truth doorposts for dataset {dataset_idx}")
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    return vis.get_picked_points()



def get_ground_truth():
    # points = np.load('data/robot_cropped_pointcloud.npy')
    # path = os.path.abspath(".")
    # path = os.path.join("")

    for i in range(0, 7):
        points = None
        if i == 0:
            # points = np.load(f"{path}/data/door0_cropped.npy")
            points = np.load(f"data/door0_cropped.npy")
        else:
            # crop margin 0.8m works decent
            # points = np.load(f"{path}/data/door{i}_cropped_m0_8.npy")
            points = np.load(f"data/door{i}_cropped_m0_8.npy")
            # crop margin 1.5m has lots of problems
            # points = np.load(f'data/door{i}_cropped_m1_5.npy')


        picked_points = pick_points(points, i)
        # response = cropped_pointcloud_to_door_post_poses_usecase(points, vis=0)
        print(
            f">>> dataset {i}: picked points = {picked_points}"
        )

if __name__ == "__main__":
    # test_pc1_success()
    # test_run()
    get_ground_truth()
