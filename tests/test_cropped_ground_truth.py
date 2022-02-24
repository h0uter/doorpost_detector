import numpy as np
import os
import pytest
from doorpost_detector.api import doorpost_pose_from_cropped_pointcloud_usecase
from doorpost_detector.utils.viz_lvl import VizLVL

def test_pointcloud1_response_success():
    # two_up =  os.path.abspath(os.path.join(__file__ ,"../.."))
    two_up = os.path.abspath(os.path.join(__file__, "../.."))
    full_path = os.path.join(two_up, "data", "door1_cropped_m0_8.npy")
    points = np.load(full_path)
    response = doorpost_pose_from_cropped_pointcloud_usecase(points, vis=VizLVL.NONE)
    assert response.success is True

def test_pointcloud0_response_poses():
    """test 10 times if the accuracy is consistent"""

    acc_x = 0.2
    acc_y = 0.2
    responses = []
    poses = []

    for i in range(5):
        two_up = os.path.abspath(os.path.join(__file__, "../.."))
        full_path = os.path.join(two_up, "data", "door0_cropped.npy")
        points = np.load(full_path)
        response = doorpost_pose_from_cropped_pointcloud_usecase(
            points, vis=VizLVL.NONE
        )
        print(
            f">>> trial {i}: success: {response.success}, poses: {response.poses}"
        )
        responses.append(response)
        poses.append(response.poses)

    print(f"variance in pose estimates {np.var(poses, axis=0)}")
    print(f"mean in pose estimates {np.mean(poses, axis=0)}")

    for response in responses:
        print(f"poses: {response.poses}")
        assert response.poses == pytest.approx(
            [0.86, -0.37, 0.93, 0.59], abs=acc_x
        )


def pointcloudN_response_poses(dataset_num, ground_truth, acc_x, acc_y):
    """test 10 times if the accuracy is consistent"""

    responses = []
    poses = []

    for i in range(5):
        two_up = os.path.abspath(os.path.join(__file__, "../.."))
        full_path = os.path.join(two_up, "data", f"door{dataset_num}_cropped_m0_8.npy")
        points = np.load(full_path)
        response = doorpost_pose_from_cropped_pointcloud_usecase(
            points, vis=VizLVL.NONE
        )
        print(
            f">>> trial {i}: success: {response.success}, poses: {response.poses}, certainty: {response.certainty}"
        )
        responses.append(response)
        poses.append(response.poses)

    print(f"variance in pose estimates {np.var(poses, axis=0)}")
    print(f"mean in pose estimates {np.mean(poses, axis=0)}")

    for response in responses:
        # print(f"poses: {response['poses']}")
        assert response.poses[0] == pytest.approx(ground_truth[0], abs=acc_x)
        assert response.poses[1] == pytest.approx(ground_truth[1], abs=acc_y)
        assert response.poses[2] == pytest.approx(ground_truth[2], abs=acc_x)
        assert response.poses[3] == pytest.approx(ground_truth[3], abs=acc_y)


def test_pointcloud1_response_poses():
    pointcloudN_response_poses(
        1, [1.84, -0.83, 1.91, 0.055], 0.2, 0.2
    )


def test_pointcloud2_response_poses():
    pointcloudN_response_poses(
        2, [1.42, -0.097, 1.33, 0.85], 0.2, 0.2
    )


# def test_pointcloud3_response_poses():
#     pointcloudN_response_poses(
#         3, [1.64, -0.76, 1.81, 0.16], 0.2, 0.2
#     )


def test_pointcloud4_response_poses():
    pointcloudN_response_poses(
        4, [1.24, -0.70, 1.37, 0.20], 0.2, 0.2
    )


def test_pointcloud5_response_poses():
    """With this pointcloud the pipeline sometimes fails to detect the door post."""
    pointcloudN_response_poses(
        5, [0.78, 1.16, -0.07, 1.40], 0.2, 0.2
    )


def test_pointcloud6_response_poses():
    pointcloudN_response_poses(
        6, [1.98, -0.27, 1.95, 0.69], 0.2, 0.2
    )





if __name__ == "__main__":
    test_pointcloud4_response_poses()
    # test_pointcloud5_response_poses()
