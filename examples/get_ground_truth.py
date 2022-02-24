import numpy as np
import os
import open3d as o3d

from doorpost_detector.utils.converters import npy2pcd


def pick_points(points: list, dataset_idx):
    pcd = npy2pcd(points)
    print("")
    print("1) Please pick at least three correspondences using [shift + left click]")
    print("   Press [shift + right click] to undo point picking")
    print("2) After picking points, press 'Q' to close the window")
    vis = o3d.visualization.VisualizerWithEditing()  # type: ignore
    vis.create_window(
        window_name=f"Pick ground truth doorposts for dataset {dataset_idx}"
    )
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    return vis.get_picked_points()


def get_ground_truth():
    ground_truth = {}

    for i in range(0, 7):
        points = None
        if i == 0:
            path = os.path.join("data", "door0_cropped.npy")
            points = np.load(path)
        else:
            # crop margin 0.8m works decent
            path = os.path.join("data", f"door{i}_cropped_m0_8.npy")

            # crop margin 1.5m has lots of problems
            # path = os.path.join("data", f"door{i}_cropped_m1_5.npy")

            points = np.load(path)

        picked_points = pick_points(points, i)

        print(f">>> dataset {i}: picked points = {points[picked_points]}")
        ground_truth[i] = points[picked_points]

    for i in range(len(ground_truth)):
        if len(ground_truth[i]) < 2:
            print(f">>> dataset {i}: ground truth not found")
        else:
            print(
                f"""
                >>> dataset {i}: ground truth = 
                {ground_truth[i][0]}
                {ground_truth[i][1]}
                """
            )

    return ground_truth


if __name__ == "__main__":
    get_ground_truth()
